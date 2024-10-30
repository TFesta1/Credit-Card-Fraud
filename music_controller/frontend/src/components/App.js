import React, { Component } from "react";
import Cohorts from "./Cohorts";
import { render } from "react-dom";

export default class App extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <Cohorts />
      </div>
    );
  }
}

const appDiv = document.getElementById("app");
render(<App />, appDiv);
