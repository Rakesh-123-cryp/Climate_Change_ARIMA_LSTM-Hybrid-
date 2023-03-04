document.addEventListener("DOMContentLoaded", () => {
    fetch('./data.json')

      .then(response => response.json())
      .then(data => {
        const chartData = data.chart_data;
        const categories = data.categories;
        
        new ApexCharts(document.querySelector("#lineChart"), {
          series: [{
            name: "Desktops",
            data: chartData
          }],
          chart: {
            height: 350,
            type: 'line',
            zoom: {
              enabled: false
            }
          },
          dataLabels: {
            enabled: false
          },
          stroke: {
            curve: 'straight'
          },
          grid: {
            row: {
              colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
              opacity: 0.5
            },
          },
          xaxis: {
            categories: categories,
          }
        }).render();
      })
      .catch(error => console.error(error));
  });