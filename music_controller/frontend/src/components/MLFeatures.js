import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Line } from "react-chartjs-2";
import BarGraph from "./BarGraph";
{
  /* <Grid container spacing={1}>
        <Grid item xs={12} align="center">
          <Typography component="h4" variant="h4">
            Create A Room
          </Typography>
        </Grid>
        <Grid item xs={12} align="center">
          <FormControl component="fieldset">
            <FormHelperText>
              <div align="center">Guest Control of Playback State</div>
            </FormHelperText>
            <RadioGroup
              row
              defaultValue="true"
              onChange={handleGuestCanPauseChange}
            >
              <FormControlLabel
                value="true"
                control={<Radio color="primary" />}
                label="Play/Pause"
                labelPlacement="bottom"
              />
              <FormControlLabel
                value="false"
                control={<Radio color="secondary" />}
                label="No Control"
                labelPlacement="bottom"
              />
            </RadioGroup>
          </FormControl>
        </Grid>
        <Grid item xs={12} align="center">
          <FormControl>
            <TextField
              required={true}
              type="number"
              onChange={handleVotesChange}
              defaultValue={defaultVotes}
              inputProps={{
                min: 1,
                style: { textAlign: "center" },
              }}
            />
            <FormHelperText>
              <div align="center">Votes Required to Skip Song</div>
            </FormHelperText>
          </FormControl>
        </Grid>
        <Grid item xs={12} align="center" onClick={handleRoomButtonPressed}>
          <Button color="primary" variant="contained">
            Create A Room
          </Button>
        </Grid>
        <Grid item xs={12} align="center">
          <Button color="secondary" variant="contained" to="/" component={Link}>
            Back
          </Button>
        </Grid>
      </Grid> */
}

const MLFeatures = () => {
  const navigate = useNavigate();
  // npm install chart.js react-chartjs-2
  const testData = [
    [65, 59, 80, 81, 56, 55, 40],
    // [28, 48, 40, 19, 86, 27, 90],
    // [18, 48, 77, 9, 100, 27, 40],
  ];
  const labels = [
    "Label1",
    "Label2",
    "Label3",
    "Label4",
    "Label5",
    "Label6",
    "Label7",
  ];

  const renderCohorts = () => {
    return (
      <div className="cohorts-container">
        <h2 className="headers">PCA Features</h2>

        {/* <h2 className="bg-dark">Cohorts Analysis</h2> */}
        <BarGraph dataSets={testData} labels={labels} />
        {/* <div className="cohorts-container-child">
          <MultiLineGraph dataSets={testData} />
        </div> */}
        {/* <MultiLineGraph dataSets={testData} />
        <MultiLineGraph dataSets={testData} /> */}
      </div>
    );
  };
  return renderCohorts();
};

export default MLFeatures;

/*
// Class based component
import React, { Component } from "react";
import Button from "@material-ui/core/Button";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import TextField from "@material-ui/core/TextField";
import FormControl from "@material-ui/core/FormControl";
import FormHelperText from "@material-ui/core/FormHelperText";
import { Radio } from "@material-ui/core";
import RadioGroup from "@material-ui/core/RadioGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import { Link, withRouter } from "react-router-dom";

export default class CreateRoomPage extends Component {
  defaultVotes = "2";

  constructor(props) {
    super(props);
    // Variables we can change later
    this.state = {
      guestCanPause: true,
      votesToSkip: this.defaultVotes,
    };
    this.handleRoomButtonPressed = this.handleRoomButtonPressed.bind(this); //Have access to the "this" keyword inside of the html below
    this.handleVotesChange = this.handleVotesChange.bind(this);
    this.handleGuestCanPauseChange = this.handleGuestCanPauseChange.bind(this);
  }

  handleVotesChange(e) {
    this.setState({
      votesToSkip: e.target.value, //get the object that called the function
    });
  }

  handleGuestCanPauseChange(e) {
    this.setState({
      guestCanPause: e.target.value === "true" ? true : false,
    });
  }

  handleRoomButtonPressed() {
    // console.log(this.state); //Has access to "this" keyword because of the bind function in the constructor
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        votes_to_skip: this.state.votesToSkip,
        guest_can_pause: this.state.guestCanPause,
      }),
    };
    fetch("/api/create-room", requestOptions)
      .then((response) => response.json())
      .then((data) => this.props.history.push("/room/" + data.code)); //Redirect to the room page
  }

  render() {
    return (
      <Grid container spacing={1}>
        <Grid item xs={12} align="center">
          <Typography component="h4" variant="h4">
            Create A Room
          </Typography>
        </Grid>
        <Grid item xs={12} align="center">
          <FormControl component="fieldset">
            <FormHelperText>
              <div align="center">Guest Control of Playback State</div>
            </FormHelperText>
            <RadioGroup
              row
              defaultValue="true"
              onChange={this.handleGuestCanPauseChange}
            >
              <FormControlLabel
                value="true"
                control={<Radio color="primary" />}
                label="Play/Pause"
                labelPlacement="bottom"
              />
              <FormControlLabel
                value="false"
                control={<Radio color="secondary" />}
                label="No Control"
                labelPlacement="bottom"
              />
            </RadioGroup>
          </FormControl>
        </Grid>
        <Grid item xs={12} align="center">
          <FormControl>
            <TextField
              required={true}
              type="number"
              onChange={this.handleVotesChange}
              defaultValue={this.defaultVotes}
              inputProps={{
                min: 1,
                style: { textAlign: "center" },
              }}
            />
            <FormHelperText>
              <div align="center">Votes Required to Skip Song</div>
            </FormHelperText>
          </FormControl>
        </Grid>
        <Grid
          item
          xs={12}
          align="center"
          onClick={this.handleRoomButtonPressed}
        >
          <Button color="primary" variant="contained">
            Create A Room
          </Button>
        </Grid>
        <Grid item xs={12} align="center">
          <Button color="secondary" variant="contained" to="/" component={Link}>
            Back
          </Button>
        </Grid>
      </Grid>
    );
  }
}
*/