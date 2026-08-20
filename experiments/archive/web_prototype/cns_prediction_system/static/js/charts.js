(function () {
    function parseData(id) {
        var el = document.getElementById(id);
        if (!el || !window.echarts) return null;
        try {
            return { el: el, data: JSON.parse(el.dataset.chart || "{}") };
        } catch (error) {
            return { el: el, data: {} };
        }
    }

    function toPercent(values) {
        return (values || []).map(function (value) {
            return value === null || value === undefined ? null : Math.round(value * 100);
        });
    }

    window.renderPatientCharts = function () {
        var risk = parseData("riskChart");
        if (risk) {
            var riskChart = echarts.init(risk.el);
            riskChart.setOption({
                tooltip: { trigger: "axis" },
                legend: { top: 0, data: ["静态风险", "动态融合风险"] },
                grid: { top: 46, left: 42, right: 20, bottom: 34 },
                xAxis: { type: "category", data: risk.data.labels || [] },
                yAxis: { type: "value", min: 0, max: 100, axisLabel: { formatter: "{value}%" } },
                series: [
                    { name: "静态风险", type: "line", smooth: true, connectNulls: true, data: toPercent(risk.data.static), itemStyle: { color: "#2563eb" } },
                    { name: "动态融合风险", type: "line", smooth: true, connectNulls: true, data: toPercent(risk.data.dynamic), itemStyle: { color: "#dc2626" } }
                ]
            });
        }

        var indicator = parseData("indicatorChart");
        if (indicator) {
            var indicatorChart = echarts.init(indicator.el);
            indicatorChart.setOption({
                tooltip: { trigger: "axis" },
                legend: { top: 0, data: ["脑脊液白细胞", "脑脊液蛋白", "脑脊液葡萄糖", "脑脊液中性粒细胞比例"] },
                grid: { top: 56, left: 48, right: 20, bottom: 34 },
                xAxis: { type: "category", data: indicator.data.labels || [] },
                yAxis: { type: "value" },
                series: [
                    { name: "脑脊液白细胞", type: "line", smooth: true, data: indicator.data.C_WBC || [] },
                    { name: "脑脊液蛋白", type: "line", smooth: true, data: indicator.data.C_P || [] },
                    { name: "脑脊液葡萄糖", type: "line", smooth: true, data: indicator.data.C_G || [] },
                    { name: "脑脊液中性粒细胞比例", type: "line", smooth: true, data: indicator.data.C_N || [] }
                ]
            });
        }

        window.addEventListener("resize", function () {
            if (risk && window.echarts.getInstanceByDom(risk.el)) window.echarts.getInstanceByDom(risk.el).resize();
            if (indicator && window.echarts.getInstanceByDom(indicator.el)) window.echarts.getInstanceByDom(indicator.el).resize();
        });
    };
})();

