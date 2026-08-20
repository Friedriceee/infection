(function () {
    document.querySelectorAll('form[data-validate="patient"]').forEach(function (form) {
        form.addEventListener("submit", function (event) {
            var age = Number(form.querySelector('[name="age"]').value);
            var patientNo = form.querySelector('[name="patient_no"]').value.trim();
            var name = form.querySelector('[name="name"]').value.trim();
            var sex = form.querySelector('[name="sex"]').value;
            if (!patientNo || !name || !sex || Number.isNaN(age) || age < 0 || age > 120) {
                event.preventDefault();
                alert("请完整填写患者编号、姓名、性别，并确认年龄在 0-120 之间。");
            }
        });
    });
})();

