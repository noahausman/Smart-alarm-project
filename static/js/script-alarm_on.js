const currentTime = document.querySelector("h1"),
scheduleBtn = document.querySelector(".ScheduleButton"),
analysisBtn = document.querySelector(".AnalysisButton");;

setInterval(() => {
    let date = new Date(),
    h = date.getHours(),
    m = date.getMinutes(),
    s = date.getSeconds(),
    ampm = "AM";
    if(h >= 12) {
        h = h - 12;
        ampm = "PM";
    }
    h = h == 0 ? h = 12 : h;
    h = h < 10 ? "0" + h : h;
    m = m < 10 ? "0" + m : m;
    s = s < 10 ? "0" + s : s;
    currentTime.innerText = `${h}:${m}:${s} ${ampm}`;
});

function scheduleAlarm() {
    document.getElementById("myForm_set_alarm").submit();
}

scheduleBtn.addEventListener("click", scheduleAlarm);

function analysisAlarm() {
    document.getElementById("myForm_analytics").submit();
}

analysisBtn.addEventListener("click", analysisAlarm);

