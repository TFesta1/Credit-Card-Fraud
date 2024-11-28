import React, { useState, useEffect, useRef, Component } from "react";
// import RoomJoinPage from "./RoomJoinPage";
import MLFeatures from "./MLFeatures";
import CreateCohorts from "./CreateCohorts";
import MLAnalysis from "./MLAnalysis";

import Room from "./Room";
import { Button, Grid, Typography, ButtonGroup } from "@material-ui/core";
import { useNavigate, Link } from "react-router-dom";

// For the router
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

/*
<Grid container spacing={3}>
        <Grid item xs={12} align="center">
          <Typography variant="h3" compact="h3">
            House Party
          </Typography>
        </Grid>
        <Grid item xs={12} align="center">
          <ButtonGroup disableElevation variant="contained" color="primary">
            <Button color="primary" to="/join" component={Link}>
              Join a Room
            </Button>
            <Button color="secondary" to="/create" component={Link}>
              Create a Room
            </Button>
          </ButtonGroup>
        </Grid>
      </Grid>
    );
  */
const HomePage = () => {
  // const navigate = useNavigate();

  const [modelFeatures, setModelFeatures] = useState([]);
  const [modelPredictions, setModelPredictions] = useState([]);
  const [modelLabels, setModelLabels] = useState([]);
  const [datasetInfo, setDatasetInfo] = useState({});
  const [tableData, setTableData] = useState([]);

  const intervalRef = useRef();
  const modelFeaturesRef = useRef(modelFeatures);
  const modelPredictionsRef = useRef(modelPredictions);
  const modelLabelsRef = useRef(modelLabels);
  const datasetInfoRef = useRef(datasetInfo);
  const [retryCount, setRetryCount] = useState(0);

  const prepareTableData = (features, predictions) => {
    // let randomIndexes = Array.from({ length: 5 }, () =>
    //   Math.floor(Math.random() * features.length)
    // );
    // randomIndexes = [100, 1000, 2000, 4000, 5000];
    // console.log("Random Indexes: ", randomIndexes);
    // const table = randomIndexes.map((row, index) => {
    //   //For each index,
    //   return {
    //     ...features[index], // Everything in the row
    //     prediction: predictions[index], // Add the prediction
    //   };
    // });
    // setTableData(table);
    // Retrieving the 5 latest ones
    const slicedFeatures = features.slice(-5); //Get the last 5 features
    const table = slicedFeatures.map((row, index) => {
      //For each index,
      return {
        ...row, // Everything in the row
        prediction: predictions[predictions.length - 1 - index], // Add the prediction
      };
    });
    setTableData(table);
    console.log("Table Data: ", table);
  };

  useEffect(() => {
    modelFeaturesRef.current = modelFeatures;
    modelPredictionsRef.current = modelPredictions;
    modelLabelsRef.current = modelLabels;
    datasetInfoRef.current = datasetInfo;
  }, [modelPredictions, modelFeatures, modelLabels, datasetInfo]);
  // npm install chart.js react-chartjs-2
  useEffect(() => {
    const fetchData = () => {
      const getRequestOptions = {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      };
      fetch("/api/get-model", getRequestOptions)
        .then((response) => response.json())
        .then((data) => {
          setModelFeatures(data[0]); //          setModel(data[0]);
          setModelPredictions(data[1]);
          setModelLabels(data[2]);
          prepareTableData(data[0], data[1]);
          const datasetInfoObj = {
            total_transactions: data[3][0],
            total_columns: data[3][1],
            total_features: data[3][2],
            total_labels: data[3][3],
            total_normal_transactions: data[3][4],
            total_fraudulent_transactions: data[3][5],
            percentage_fraudulent: data[3][6],
            percentage_normal: data[3][7],
          };
          setDatasetInfo(datasetInfoObj);
        })
        .catch((error) => {
          console.error("Error fetching model: ", error);
          setRetryCount((prevCount) => prevCount + 1);
        });
    };
    fetchData();
    // We could easily modify this code to just run every 3s if we needed a script to do so, but we're running it once here
    // modelPredictionsRef.current.length === 0 ||
    // modelLabelsRef.current.length === 0
    intervalRef.current = setInterval(() => {
      if (
        modelFeaturesRef.current.length === 0 ||
        modelPredictionsRef.current.length === 0 ||
        modelLabelsRef.current.length === 0 ||
        datasetInfoRef.current.length === 0
      ) {
        fetchData();
        console.log("Retrying to fetch model data: ", modelFeatures);
      } else {
        console.log(
          " Model Features ",
          modelFeaturesRef,
          " Model Predictions ",
          modelPredictionsRef,
          " Model Labels ",
          modelLabelsRef,
          " Dataset Info ",
          datasetInfoRef
        );
        clearInterval(intervalRef.current);
      }
    }, 500); // Retry every 3 seconds
  }, []); //This means it runs when either model or modelData change

  const renderHomePage = () => {
    return (
      <div className="home-container center">
        <nav className="navbar navbar-expand-lg navbar-light bg-light">
          <Link className="navbar-brand" to="/">
            Options
          </Link>
          <button
            className="navbar-toggler"
            type="button"
            data-toggle="collapse"
            data-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ml-auto">
              {/* Nav links */}
              <li className="nav-item">
                <Link className="nav-link" to="/mlAnalysis">
                  Model Analysis
                </Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/cohorts">
                  Cohorts
                </Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/mlFeatures">
                  Machine Learning Features
                </Link>
              </li>
            </ul>
          </div>
        </nav>
        <div>
          <h2 className="headers">Latest Feed</h2>
        </div>
        {/* Table selection */}
        <div className="container mt-5">
          <div className="table-responsive">
            <table className="table table-bordered">
              <thead>
                <tr>
                  {/* <th>Customer ID</th>
                  <th>Customer Name</th>
                  <th>Transaction Amount</th>
                  <th>Fraud?</th> */}
                  <th>Prediction</th>
                  {modelLabelsRef.current.map((label, index) => (
                    <th key={index}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <td
                      className={
                        row.prediction === 1 ? "fraudulent" : "non-fraudulent"
                      }
                    >
                      {row.prediction === 1 ? "Fraudulent" : "Non-Fraudulent"}
                    </td>
                    {Object.values(row)
                      .slice(0, -1) //0 to everything besides the last element
                      .map((value, colIndex) => (
                        <td key={colIndex}>{value}</td>
                      ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h2 className="headers">Important Dataset Info</h2>
        </div>
        <div className="info-section">
          <p>
            <span className="label">Total Transactions:</span>{" "}
            {datasetInfo.total_transactions}
          </p>
          <p>
            <span className="label">Total Columns:</span>{" "}
            {datasetInfo.total_columns}
          </p>
          <p>
            <span className="label">Total Features:</span>{" "}
            {datasetInfo.total_features}
          </p>
          <p>
            <span className="label">Total Labels:</span>{" "}
            {datasetInfo.total_labels}
          </p>
          <p>
            <span className="label">Total Normal Transactions:</span>{" "}
            {datasetInfo.total_normal_transactions}
          </p>
          <p>
            <span className="label">Total Fraudulent Transactions:</span>{" "}
            {datasetInfo.total_fraudulent_transactions}
          </p>
          <p>
            <span className="label">Percentage Fraudulent:</span>{" "}
            {datasetInfo.percentage_fraudulent}
          </p>
          <p>
            <span className="label">Percentage Normal:</span>{" "}
            {datasetInfo.percentage_normal}
          </p>
        </div>
      </div>
    );
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={renderHomePage()} />
        <Route path="/mlFeatures" element={<MLFeatures />} />
        <Route path="/cohorts" element={<CreateCohorts />} />
        <Route path="/mlAnalysis" element={<MLAnalysis />} />
        <Route path="/room/:roomCode" element={<Room />} />
        {/* :roomCode is a variable. This by default passes roomCode as "matched", which is just a param for how it got there, and we can use this to get the room */}
      </Routes>
    </Router>
  );
};

export default HomePage;
