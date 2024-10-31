import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale, // Scales X axis
  LinearScale, // Scales Y axis
  PointElement, //Plotting
  LineElement, // Lines between points
  Title,
  Tooltip,
  Legend
);

//dataSets is the prop
const MultiLineGraph = ({ dataSets }) => {
  const data = {
    labels: Array.from(
      //labels made from the max length of the dataSets
      { length: Math.max(...dataSets.map((data) => data.length)) },
      (_, i) => `P${i + 1}`
    ),
    datasets: dataSets.map((data, index) => ({
      label: `Cohort ${index + 1}`,
      data, //datapoints
      fill: false, //area under line filled
      borderColor: `rgba(${index * 50}, 99, 132, 1)`, //color of the line
      tension: 0.1, //curve of the line
    })),
  };

  return <Line data={data} />;
};

export default MultiLineGraph;
