import React from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const BarGraph = ({ dataSets, labels }) => {
  const colors = [
    "rgba(54, 162, 235, 0.5)",
    "rgba(75, 192, 192, 0.5)",
    "rgba(153, 102, 255, 0.5)",
    "rgba(255, 159, 64, 0.5)",
  ];
  const borderColors = [
    "rgba(255, 99, 132, 1)",
    "rgba(54, 162, 235, 1)",
    "rgba(75, 192, 192, 1)",
    "rgba(153, 102, 255, 1)",
    "rgba(255, 159, 64, 1)",
  ];

  const data = {
    labels: labels,

    // Array.from(
    //   { length: Math.max(...labels.map((data) => data.length)) },
    //   (label, i) => `${labels[i]}`
    // ),

    datasets: dataSets.map((data, index) => ({
      label: "Features", //`Cohort ${index + 1}`,
      data,
      fill: false,
      backgroundColor: data.map((_, i) => colors[i % colors.length]), // i % colors.length means division remainder is the index
      borderColor: data.map((_, i) => borderColors[i % colors.length]), //`rgba(${index * 125 + 50}, 99, 132, 1)`,
      borderWidth: 1,
    })),
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Uncorrelated Variables",
      },
    },
  };

  return <Bar data={data} options={options} />;
};

export default BarGraph;
