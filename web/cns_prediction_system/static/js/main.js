(function () {
    document.querySelectorAll(".flash").forEach(function (el) {
        window.setTimeout(function () {
            el.style.opacity = "0.25";
        }, 4500);
    });

    document.querySelectorAll("form[data-confirm]").forEach(function (form) {
        form.addEventListener("submit", function (event) {
            if (!window.confirm(form.getAttribute("data-confirm"))) {
                event.preventDefault();
            }
        });
    });
})();
