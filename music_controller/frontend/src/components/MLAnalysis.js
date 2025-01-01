import React, { useState, useEffect, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";

const MLAnalysis = () => {
  const [modelImages, setModelImages] = useState([]);
  const [modelDetails, setModelDetails] = useState([]);
  const [modelHeaders, setModelHeaders] = useState([]);
  const [modelStats, setModelStats] = useState([]);

  const intervalRef = useRef();
  const modelDetailsRef = useRef(modelDetails);
  const modelStatsRef = useRef(modelStats);
  const modelHeadersRef = useRef(modelHeaders);
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

  const parseDataHeaders = (dataString) => {
    //For data like item1   item2    item3\n... (splits by 2 or more spaces, then makes it an object)
    const rows = dataString.trim().split("\n");
    const headers = rows[0].trim().split(/\s{2,}/);
    return headers;
  };

  const parseDataStringRows = (dataString) => {
    //For data like item1   item2    item3\n... (splits by 2 or more spaces, then makes it an object)
    const rows = dataString.trim().split("\n");
    const headers = rows[0].trim().split(/\s{2,}/);
    const data = rows.slice(1).map((row) => {
      // skip the first row
      const values = row.trim().split(/\s{2,}/);
      return headers.reduce((obj, header, index) => {
        // reduce to an object
        obj[header.trim()] = values[index].trim();
        return obj;
      }, {});
    });
    console.log("Data: ", data);
    return data;
  };

  useEffect(() => {
    modelDetailsRef.current = modelDetails;
    console.log("Model Details Ref0: ", modelDetailsRef["current"]);
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
          setModelHeaders(data[4]);
          setModelStats(data[5]);
          // console.log(`modelDetails: ${parseDataString(data[5])}`);
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
        console.log("Model Details ", modelDetailsRef[0]);
        console.log("Model Details1 ", modelDetailsRef[1]);
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
          {modelDetails.length > 0 ? (
            <table className="table">
              <thead classNAme="thead-dark">
                <tr>
                  {Object.keys(modelDetails[1][0]).map((header, index) => (
                    <th key={index}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelDetails.slice(1).map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {Object.values(row).map(
                      (value, colIndex) =>
                        Object.keys(value).map((header, i) => (
                          <td key={colIndex}>
                            {i}
                            {/* value[header] */}
                            {/* {typeof value === "object"
                            ? JSON.stringify(value)
                            : String(value)} */}
                          </td>
                        ))
                      // <td key={colIndex}>
                      //   {value["Model"]}
                      //   {/* {typeof value === "object"
                      //     ? JSON.stringify(value)
                      //     : String(value)} */}
                      // </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div>Loading...</div>
          )}
        </div>
        <div>
          <h2 className="headers">Various Model Metrics</h2>
        </div>
        {/* <div>
          {modelImages.map((modelImage, i) => (
            <div key={modelImage.id}>
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
        </div> */}
      </div>
    );
  };
  return renderAnalysis();
};

export default MLAnalysis;
