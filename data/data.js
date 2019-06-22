const encoder = require('wav-encoder');
const decoder = require('wav-decoder');
const peaking = require('node-peaking');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const _ = require('lodash');
const config = require('../config');


class Data {
  constructor(args) {
    this.samplerate = 44100;
    this.shortTimeSamples = 2048 // Math.pow(2, 16);

    this.sounds = [];
    this.filterParams = [];
    this.dataSets = [];

    this.index = 0;
    this.batchIndex = 0;
  }

  randomFilterParam() {
    const params = {
      freq: _.random(20, 20000),
      q: _.random(0, 5.0, true),
      gain: _.random(-15.0, 15.0, true),
      bw: _.random(0, 5.0, true),
      samplerate: this.samplerate
    }
    return params;
  }

  async separateSound(soundPath, chunnel = 0) {
    const soundFilaPeth = path.resolve(soundPath);
    let soundBuffer = await decoder.decode(fs.readFileSync(soundFilaPeth));
    soundBuffer = soundBuffer.channelData[chunnel];

    const shortTimeSounds = _.chunk(soundBuffer, this.shortTimeSamples);
    shortTimeSounds.pop();

    return shortTimeSounds;
  }

  async separation() {
    for (let i = 0; i < config.numSamples; i++) {
      let p = path.join(__dirname, 'sounds', i + '.wav');
      let buffers = await this.separateSound(p, 0);
      // random selection
      let index = 0;
      let randIndex = Array(buffers.length).fill(0);
      randIndex = randIndex.map((elem) => {
        return index++;
      });
      randIndex = _.shuffle(randIndex).slice(0, config.samplesPerSong);
      let randL = [];
      for (let i = 0; i < randIndex.length; i++) {
        randL.push(buffers[randIndex[i]]);
      }

      if (config.varbose) console.log(i + ' / ' + config.numSamples);

      this.sounds.push(...randL);
    }
  }

  applyFilter() {
    for (let i = 0; i < this.sounds.length; i++) {
      const filterParam = this.randomFilterParam();
      const filtered = peaking.peaking(this.sounds[i], filterParam);

      this.sounds[i] = filtered;
      this.filterParams.push(filterParam);
    }
  }

  makeDataset() {
    for (let n = 0; n < config.trainEpoches; n++) {
      const xBatch = [];
      const labelBatch = [];
      for (let i = 0; i < config.batchSize; i++) {
        xBatch.push(Array(...this.sounds[i]));
        labelBatch.push([
          this.filterParams[i].freq / 20000,
          this.filterParams[i].q / 5.0,
          this.filterParams[i].gain / 15.0,
          this.filterParams[i].bw / 5.0
        ]);
      }
      this.dataSets.push([xBatch, labelBatch]);
    }
    this.dataSets = _.shuffle(this.dataSets);
  }

  nextBatch() {
    this.loadDataset(this.index);
    this.index++;
    return {
      xs: tf.tensor(this.dataSets[0]),
      ys: tf.tensor(this.dataSets[1])
    }
  }

  loadDataset(index) {
    const filePath = config.dataSetPath + index + '.json';
    let json = JSON.parse(fs.readFileSync(filePath));
    this.dataSets = json;
  }

  disposer(tensors) {
    tensors.forEach((elem) => {
      elem.dispose();
    });
  }
}

module.exports = Data;
