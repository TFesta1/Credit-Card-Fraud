import React, { useState, useEffect, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";

const MLAnalysis = () => {
  const [modelImages, setModelImages] = useState([]);
  useEffect(() => {
    const fetchData = () => {
      const getRequestOptions = {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      };
      fetch("/api/get-model-images", getRequestOptions)
        .then((response) => response.json())
        .then((data) => {
          console.log(`data: ${JSON.stringify(data)}`);
          setModelImages(data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    };
    fetchData();
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
              <h3>{modelImage.name}</h3>
              <img
                src={modelImage.path}
                alt={modelImage.name}
                className="model-image"
              />
            </div>
          ))}
        </div>
      </div>
    );
  };
  return renderAnalysis();
};

export default MLAnalysis;
