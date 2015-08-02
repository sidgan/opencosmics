
google.setOnLoadCallback(drawChart);
  function drawChart() {
    var data = google.visualization.arrayToDataTable([
      ['Mon', 9, 8, 3, 4],
      ['Tue', 8, 8, 5, 6],
      ['Wed', 5, 5, 7, 8],
      ['Thu', 7, 7, 6, 5],
      ['Fri', 8, 6, 2, 5]
      // Treat first row as data as well.
    ], true);

    var options = {
      legend:'none'
    };

    var chart = new google.visualization.CandlestickChart(document.getElementById('chart_div'));

    chart.draw(data, options);
  }


