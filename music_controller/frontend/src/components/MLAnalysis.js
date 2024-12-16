import React, { useState, useEffect, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";

const MLAnalysis = () => {
  const [modelImages, setModelImages] = useState([]);
  const [modelDetails, setModelDetails] = useState([]);

  const intervalRef = useRef();
  const modelDetailsRef = useRef(modelDetails);
  const [retryCount, setRetryCount] = useState(0);

  const parseDataString = (dataString) => {
    //For data like item1   item2    item3\n... (splits by 2 or more spaces, then makes it an object)
    const rows = dataString.trim().split("\n");
    const headers = rows[0].trim().split(/\s{2,}/); // split by 2 or more spaces
    const data = rows.slice(1).map((row) => {
      // skip the first row
      const values = row.trim().split(/\s{2,}/);
      return headers.reduce((obj, header, index) => {
        // reduce to an object
        obj[header.trim()] = values[index].trim();
        return obj;
      }, {});
    });
    return data;
  };

  useEffect(() => {
    modelDetailsRef.current = modelDetails;
    console.log("Model Details Ref: ", modelDetailsRef);
  }, [modelDetails]);

  useEffect(() => {
    const fetchData = () => {
      const getRequestOptions = {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      };
      fetch("/api/get-model-images", getRequestOptions)
        .then((response) => response.json())
        .then((data) => {
          // console.log(`data: ${JSON.stringify(data)}`);
          setModelImages(data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    };
    fetchData();
  }, []);

  useEffect(() => {
    const fetchData = () => {
      const getRequestOptions = {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      };
      fetch("/api/get-model", getRequestOptions)
        .then((response) => response.json())
        .then((data) => {
          setModelDetails([data[4], parseDataString(data[5])]);
          console.log(`modelDetails: ${parseDataString(data[5])}`);
          // console.log(`data: `);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    };
    fetchData();

    intervalRef.current = setInterval(() => {
      if (modelDetailsRef.current.length === 0) {
        fetchData();
        console.log("Retrying to fetch model data: ", modelDetailsRef);
      } else {
        console.log("Model Details ", modelDetailsRef);
        clearInterval(intervalRef.current);
      }
    }, 500);
  }, []);

  const renderAnalysis = () => {
    return (
      <div className="home-container center width-customPages">
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
                <Link className="nav-link" to="/">
                  Home
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
          <h2 className="headers">Various Model Metrics</h2>
        </div>
        <div>
          {modelImages.map((modelImage, i) => (
            <div key={modelImage.id}>
              {/* <h3>{modelImage.name}</h3> */}
              <img
                src={modelImage.path}
                alt={modelImage.name}
                className="model-image"
              />
            </div>
          ))}
        </div>
        <div>
          <h2 className="headers">Model Comparisons</h2>
        </div>
        <div>
          <h3 className="headers">{modelDetailsRef[1][0]}</h3>
          {/* {modelDetails.length > 0 ? (
            <table>
              <thead>
                <tr>
                  {modelDetails[1][0].map((header, index) => (
                    <th key={index}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelDetails[1].map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {modelDetails[1][0].map((header, colIndex) => (
                      <td key={colIndex}>{row[header]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div> Loading... </div>
          )} */}
        </div>
      </div>
    );
  };
  return renderAnalysis();
};

export default MLAnalysis;
