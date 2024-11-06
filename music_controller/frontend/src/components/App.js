import React, { Component } from "react";
import HomePage from "./HomePage";
import RoomJoinPage from "./RoomJoinPage";
import CreateRoomPage from "./CreateCohorts";

import { render } from "react-dom";

export default class App extends Component {
  constructor(props) {
    super(props);
  }
  //className="center"
  render() {
    return (
      <div className="width-100">
        <HomePage />
      </div>
    );
  }
}

const appDiv = document.getElementById("app");
if (appDiv) {
  render(<App />, appDiv);
} else {
  console.error("App div not found!");
}
// render(<App />, appDiv);
