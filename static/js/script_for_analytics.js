const scheduleBtn = document.querySelector("button");

function scheduleAlarm() {
    document.getElementById("myForm").submit();
}

scheduleBtn.addEventListener("click", scheduleAlarm);
