/* Project 2 — Explainability dashboard front-end.
 *
 * Responsibilities:
 *   - hold the current (modelType, lambda) state;
 *   - keep the lambda slider, model summary, model view (tree/coefficients),
 *     counterfactuals and feature-effect plots all in sync;
 *   - talk to the Django JSON API and render everything with Plotly.
 *
 * Everything is plain ES6 + Plotly (loaded via CDN in the template). No build
 * step, matching the spirit of the skeleton.
 */
(function () {
    "use strict";

    // Consistent per-species colours used across every plot.
    const SPECIES_COLORS = {
        Adelie: "#275CB2",
        Chinstrap: "#E07B39",
        Gentoo: "#2E8B72",
    };
    const colorFor = (cls, i) =>
        SPECIES_COLORS[cls] || ["#275CB2", "#E07B39", "#2E8B72", "#8B5CF6"][i % 4];

    const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };
    const FONT = { family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", size: 13 };

    const state = {
        modelType: "tree",
        lambda: 0,
        meta: null, // filled from /api/meta
    };

    // --- tiny helpers ------------------------------------------------------
    const $ = (id) => document.getElementById(id);

    function debounce(fn, ms) {
        let t = null;
        return function (...args) {
            clearTimeout(t);
            t = setTimeout(() => fn.apply(this, args), ms);
        };
    }

    function getJSON(url, params) {
        const qs = params ? "?" + new URLSearchParams(params).toString() : "";
        return fetch(url + qs, { headers: { Accept: "application/json" } }).then((r) => {
            if (!r.ok) throw new Error("Request failed: " + r.status);
            return r.json();
        });
    }

    function postJSON(url, body) {
        return fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": window.P2_CSRF,
                Accept: "application/json",
            },
            body: JSON.stringify(body),
        }).then((r) => {
            if (!r.ok) throw new Error("Request failed: " + r.status);
            return r.json();
        });
    }

    // ----------------------------------------------------------------------
    // Initialisation
    // ----------------------------------------------------------------------
    function init() {
        getJSON(window.P2_URLS.meta).then((meta) => {
            state.meta = meta;
            populateInstances(meta.instances);
            populateTargets(meta.classes);
            applyLambdaRange();
            wireControls();
            refreshAll();
        });
    }

    function populateInstances(instances) {
        const sel = $("cf-instance");
        sel.innerHTML = "";
        instances.forEach((inst) => {
            const o = document.createElement("option");
            o.value = inst.index;
            o.textContent = inst.label;
            sel.appendChild(o);
        });
    }

    function populateTargets(classes) {
        const sel = $("cf-target");
        sel.innerHTML = "";
        classes.forEach((c) => {
            const o = document.createElement("option");
            o.value = c;
            o.textContent = c;
            sel.appendChild(o);
        });
    }

    function applyLambdaRange() {
        const m = state.meta[state.modelType];
        const slider = $("lambda-slider");
        slider.min = m.lambda_min;
        slider.max = m.lambda_max;
        slider.step = (m.lambda_max - m.lambda_min) / 200 || 0.0001;
        // keep current lambda within range
        state.lambda = Math.min(Math.max(state.lambda, m.lambda_min), m.lambda_max);
        slider.value = state.lambda;
        $("lambda-value").textContent = state.lambda.toFixed(4);
        $("lambda-hint").textContent =
            "(0 = most accurate \u2192 " + m.lambda_max.toFixed(3) + " = simplest)";
    }

    function wireControls() {
        // model-type segmented buttons
        document.querySelectorAll("#model-type-toggle .p2-seg-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                document
                    .querySelectorAll("#model-type-toggle .p2-seg-btn")
                    .forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                state.modelType = btn.dataset.value;
                applyLambdaRange();
                refreshAll();
            });
        });

        // lambda slider — live label, debounced network refresh
        const slider = $("lambda-slider");
        slider.addEventListener("input", () => {
            state.lambda = parseFloat(slider.value);
            $("lambda-value").textContent = state.lambda.toFixed(4);
            debouncedRefreshModelAndEffects();
        });

        // feature-effect feature selector
        $("fe-feature").addEventListener("change", refreshFeatureEffect);

        // counterfactual generate button
        $("cf-run").addEventListener("click", runCounterfactuals);

        // "record this model" button -> persist current selection
        $("record-run").addEventListener("click", recordRun);
    }

    const debouncedRefreshModelAndEffects = debounce(() => {
        refreshModel();
        refreshFeatureEffect();
    }, 180);

    function refreshAll() {
        refreshModel();
        refreshFeatureEffect();
        // counterfactuals are only recomputed on demand (they sample), but we
        // clear stale output when the model changes.
        $("cf-result").innerHTML =
            '<p class="p2-placeholder">Pick an instance and target class, then press Generate.</p>';
    }

    // ----------------------------------------------------------------------
    // Model panel (Tasks 1–3)
    // ----------------------------------------------------------------------
    function refreshModel() {
        getJSON(window.P2_URLS.model, {
            model_type: state.modelType,
            lambda: state.lambda,
        }).then((data) => {
            // summary chips
            $("chip-acc").textContent = (data.acc_test * 100).toFixed(1) + "%";
            $("chip-acc-train").textContent = (data.acc_train * 100).toFixed(1) + "%";
            $("chip-omega").textContent = data.omega;
            $("chip-omega-label").innerHTML =
                "&Omega; (" + data.omega_label + ")";

            drawSelectionCurve(data.selection_curve);

            if (data.model_type === "tree") {
                $("tree-container").style.display = "";
                $("coef-container").style.display = "none";
                drawTree(data.tree);
                $("tree-text").textContent = data.tree_text;
            } else {
                $("tree-container").style.display = "none";
                $("coef-container").style.display = "";
                drawCoefficients(data.coefficients);
            }
        });
    }

    // Accuracy-vs-complexity curve with the currently selected model marked.
    function drawSelectionCurve(curve) {
        const x = curve.map((d) => d.omega);
        const acc = curve.map((d) => d.acc_test);
        const sel = curve.find((d) => d.selected);

        const traces = [
            {
                x: x,
                y: acc,
                mode: "lines+markers",
                name: "Test accuracy",
                line: { color: "#275CB2", width: 2 },
                marker: { size: 7 },
                hovertemplate: "Ω=%{x}<br>acc=%{y:.3f}<extra></extra>",
            },
            {
                x: [sel.omega],
                y: [sel.acc_test],
                mode: "markers",
                name: "Selected (λ)",
                marker: { size: 16, color: "#E07B39", symbol: "circle-open", line: { width: 3 } },
                hovertemplate: "selected<br>Ω=%{x}<br>acc=%{y:.3f}<extra></extra>",
            },
        ];

        Plotly.react(
            "selection-plot",
            traces,
            {
                title: { text: "Accuracy vs. complexity (Ω)", font: { size: 14 } },
                margin: { t: 36, r: 10, b: 40, l: 50 },
                xaxis: { title: "Complexity Ω" },
                yaxis: { title: "Test accuracy", rangemode: "tozero" },
                font: FONT,
                showlegend: true,
                legend: { orientation: "h", y: -0.25 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
            },
            PLOTLY_CONFIG
        );
    }

    // ---- custom decision-tree drawing as a Plotly scatter + shapes --------
    function drawTree(root) {
        // assign x positions by in-order traversal of leaves, y by depth.
        const nodes = [];
        const edges = [];
        let leafCounter = 0;

        function layout(node, depth) {
            node._depth = depth;
            if (node.is_leaf) {
                node._x = leafCounter++;
            } else {
                layout(node.left, depth + 1);
                layout(node.right, depth + 1);
                node._x = (node.left._x + node.right._x) / 2;
                edges.push([node, node.left]);
                edges.push([node, node.right]);
            }
            node._y = -depth;
            nodes.push(node);
        }
        layout(root, 0);

        const shapes = edges.map(([a, b]) => ({
            type: "line",
            x0: a._x,
            y0: a._y,
            x1: b._x,
            y1: b._y,
            line: { color: "#A1B4D4", width: 1.5 },
        }));

        const nodeX = [],
            nodeY = [],
            text = [],
            hover = [],
            colors = [];
        nodes.forEach((n) => {
            nodeX.push(n._x);
            nodeY.push(n._y);
            if (n.is_leaf) {
                text.push(n.class + "<br>n=" + n.n_samples);
                colors.push(colorFor(n.class, 0));
                hover.push(
                    n.class +
                        "<br>samples: " +
                        n.n_samples +
                        "<br>p: " +
                        n.proba.map((p) => p.toFixed(2)).join(", ")
                );
            } else {
                text.push(n.feature_label + "<br>≤ " + n.threshold.toFixed(1));
                colors.push("#ffffff");
                hover.push(
                    "split: " +
                        n.feature_label +
                        " ≤ " +
                        n.threshold.toFixed(2) +
                        "<br>samples: " +
                        n.n_samples
                );
            }
        });

        const trace = {
            x: nodeX,
            y: nodeY,
            mode: "markers+text",
            type: "scatter",
            text: text,
            textposition: "middle center",
            textfont: { size: 10, color: "#173a6b", family: FONT.family },
            hovertext: hover,
            hoverinfo: "text",
            marker: {
                size: 58,
                color: colors,
                line: { color: "#275CB2", width: 1.5 },
                symbol: "square",
            },
            cliponaxis: false,
        };

        const maxLeaf = Math.max(...nodes.map((n) => n._x));
        const maxDepth = Math.max(...nodes.map((n) => n._depth));

        Plotly.react(
            "tree-plot",
            [trace],
            {
                title: { text: "Decision tree", font: { size: 14 } },
                shapes: shapes,
                margin: { t: 36, r: 30, b: 20, l: 30 },
                xaxis: { visible: false, range: [-0.8, maxLeaf + 0.8] },
                yaxis: { visible: false, range: [-(maxDepth + 0.6), 0.6] },
                font: FONT,
                height: 150 + maxDepth * 110,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
            },
            PLOTLY_CONFIG
        );
    }

    // ---- logistic-regression coefficients as a grouped bar chart ----------
    function drawCoefficients(coefs) {
        const classes = Object.keys(coefs);
        const features = Object.keys(coefs[classes[0]]);
        const traces = classes.map((cls, i) => ({
            x: features,
            y: features.map((f) => coefs[cls][f]),
            name: cls,
            type: "bar",
            marker: { color: colorFor(cls, i) },
        }));

        Plotly.react(
            "coef-container",
            traces,
            {
                title: {
                    text: "Logistic-regression coefficients (standardised features)",
                    font: { size: 14 },
                },
                barmode: "group",
                margin: { t: 40, r: 10, b: 110, l: 50 },
                xaxis: { tickangle: -40 },
                yaxis: { title: "Weight", zeroline: true },
                font: FONT,
                legend: { orientation: "h", y: -0.45 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
            },
            PLOTLY_CONFIG
        );
    }

    // ----------------------------------------------------------------------
    // Counterfactuals (Task 4)
    // ----------------------------------------------------------------------
    function runCounterfactuals() {
        const btn = $("cf-run");
        btn.disabled = true;
        btn.textContent = "Searching…";
        $("cf-result").innerHTML =
            '<p class="p2-placeholder">Sampling locally and ranking by MAD-weighted L1 distance…</p>';

        postJSON(window.P2_URLS.counterfactuals, {
            model_type: state.modelType,
            lambda: state.lambda,
            instance_index: parseInt($("cf-instance").value, 10),
            target_label: $("cf-target").value,
            k: parseInt($("cf-k").value, 10),
        })
            .then(renderCounterfactuals)
            .catch(() => {
                $("cf-result").innerHTML =
                    '<p class="p2-error">Something went wrong generating counterfactuals.</p>';
            })
            .finally(() => {
                btn.disabled = false;
                btn.textContent = "Generate";
            });
    }

    function renderCounterfactuals(res) {
        const out = $("cf-result");
        const orig = res.original_instance;
        const origHtml =
            '<div class="p2-cf-orig"><strong>Original</strong> (predicted ' +
            res.original_label +
            "): " +
            Object.entries(orig)
                .map(([k, v]) => `<span class="p2-kv">${k}: ${v}</span>`)
                .join(" ") +
            "</div>";

        if (!res.found) {
            out.innerHTML =
                origHtml +
                '<p class="p2-error">No counterfactual for target <strong>' +
                res.target_label +
                "</strong> found, even after widening the search (sampled " +
                res.n_sampled +
                " points). This often means the target class is unreachable by small local changes from this instance under the current model.</p>";
            return;
        }

        let html = origHtml;
        html +=
            '<p class="p2-cf-caption">Closest changes that make the model predict <strong>' +
            res.target_label +
            "</strong>:</p>";

        res.counterfactuals.forEach((cf, i) => {
            const rows = cf.changes
                .map((c) => {
                    if (c.delta === null) {
                        return `<tr><td>${c.feature}</td><td>${c.from}</td><td>→</td><td><strong>${c.to}</strong></td><td class="p2-cat">category</td></tr>`;
                    }
                    const dir = c.delta > 0 ? "▲" : "▼";
                    const cls = c.delta > 0 ? "p2-up" : "p2-down";
                    return `<tr><td>${c.feature}</td><td>${c.from}</td><td>→</td><td><strong>${c.to}</strong></td><td class="${cls}">${dir} ${Math.abs(
                        c.delta
                    ).toFixed(2)}</td></tr>`;
                })
                .join("");
            html +=
                '<div class="p2-cf-card"><div class="p2-cf-rank">CF #' +
                (i + 1) +
                ' <span class="p2-cf-dist">distance ' +
                cf.distance +
                "</span></div>" +
                '<table class="p2-cf-table"><thead><tr><th>Feature</th><th>from</th><th></th><th>to</th><th>change</th></tr></thead><tbody>' +
                rows +
                "</tbody></table></div>";
        });

        out.innerHTML = html;
    }

    // ----------------------------------------------------------------------
    // Feature effects: PDP + ALE (Task 5)
    // ----------------------------------------------------------------------
    function refreshFeatureEffect() {
        getJSON(window.P2_URLS.featureEffect, {
            model_type: state.modelType,
            lambda: state.lambda,
            feature: $("fe-feature").value,
        }).then((data) => {
            drawEffect("pdp-plot", data.pdp.grid, data.pdp, "Partial Dependence (PDP)", "Mean predicted probability");
            drawEffect(
                "ale-plot",
                edgesToCenters(data.ale.edges),
                data.ale,
                "Accumulated Local Effects (ALE)",
                "Centered effect on probability"
            );
            $("ale-method-note").innerHTML =
                "ALE computed from scratch using the <strong>" +
                data.ale.method +
                "</strong> method for the " +
                (data.model_type === "logistic" ? "logistic-regression" : "decision-tree") +
                " model. PDP is the Monte-Carlo average over all " +
                "background rows. Each plot shows one curve per species.";
        });
    }

    function edgesToCenters(edges) {
        // ALE values live at the bin edges; for plotting against x we keep the
        // edges themselves (one accumulated value per edge).
        return edges;
    }

    function drawEffect(divId, xvals, payload, title, ytitle) {
        const traces = payload.classes.map((cls, i) => ({
            x: xvals,
            y: payload.curves[cls],
            mode: "lines",
            name: cls,
            line: { color: colorFor(cls, i), width: 2.5, shape: "spline" },
        }));

        Plotly.react(
            divId,
            traces,
            {
                title: { text: title, font: { size: 14 } },
                margin: { t: 40, r: 10, b: 50, l: 55 },
                xaxis: { title: payload.feature_label },
                yaxis: { title: ytitle },
                font: FONT,
                legend: { orientation: "h", y: -0.3 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
            },
            PLOTLY_CONFIG
        );
    }

    // ----------------------------------------------------------------------
    // Recorded run history (persisted server-side with the Django ORM)
    // ----------------------------------------------------------------------
    function recordRun() {
        const btn = $("record-run");
        const original = btn.innerHTML;
        btn.disabled = true;
        btn.textContent = "Recording…";

        postJSON(window.P2_URLS.record, {
            model_type: state.modelType,
            lambda: state.lambda,
        })
            .then((payload) => {
                renderHistory(payload);
                btn.innerHTML = "✓ Recorded";
                setTimeout(() => {
                    btn.innerHTML = original;
                    btn.disabled = false;
                }, 1200);
            })
            .catch(() => {
                btn.innerHTML = original;
                btn.disabled = false;
            });
    }

    function renderHistory(payload) {
        $("hist-total").textContent = payload.total_runs;
        const body = $("history-body");

        if (!payload.history.length) {
            body.innerHTML =
                '<tr id="history-empty"><td colspan="7" class="p2-history-empty">' +
                "No models recorded yet.</td></tr>";
            return;
        }

        body.innerHTML = payload.history
            .map((r) => {
                const best =
                    r.id === payload.best_id ? ' class="p2-history-best"' : "";
                const omega = Number.isInteger(r.omega)
                    ? r.omega
                    : r.omega.toFixed(2);
                return (
                    '<tr data-run-id="' + r.id + '"' + best + ">" +
                    "<td>" + r.created_at + "</td>" +
                    "<td>" + r.model_label + "</td>" +
                    "<td>" + r.lam.toFixed(4) + "</td>" +
                    "<td>" + omega + " <small>" + r.omega_label + "</small></td>" +
                    "<td>" + (r.hyperparam || "") + "</td>" +
                    "<td>" + r.acc_test.toFixed(4) + "</td>" +
                    "<td>" + r.acc_train.toFixed(4) + "</td>" +
                    "</tr>"
                );
            })
            .join("");
    }

    // go
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
