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
    this.dataSets = [];

    this.index = 0;
    this.batchIndex = 0;
  }

  randomFilterParam() {
    const params = {
      freq: _.random(20, 20000),
      q: _.random(0, 5.0, true),
      gain: _.random(-15.0, 15.0, true),
      bw: _.random(0, 5, true),
      samplerate: this.samplerate
    }
    return params;
  }

  chunk(buffer) {
    let chunks = [];
    let chunkHop = [];
    let chunkOverwrap = [];
    let chunkIndex = 0;
    for (let i = 0; i < buffer.length; i++) {
      if (chunkIndex < (this.shortTimeSamples / 2)) {
        chunkHop.push(buffer[i]);
        chunkOverwrap.push(buffer[i]);
        chunkIndex++;
      }
      else {
        if (i !== this.shortTimeSamples / 2) chunks.push(chunkOverwrap);
        chunkOverwrap = chunkHop.slice(0, chunkHop.length);
        chunkHop = [];
        chunkIndex = 0;
      }
    }
    chunks.push(chunkOverwrap);

    chunks = chunks.map((chunk) => {
      return this.window(chunk);
    });
    return chunks;
  }

  window(buffer) {
    const window = [];
    for (let i = 0; i < buffer.length; i++) {
      const x = i / buffer.length;
      window.push(buffer[i] * 0.5 * (1.0 - Math.cos(2.0 * 3.141592 * x))); // hanning window
    }
    return window;
  }

  async separateSound(soundPath, chunnel = 0) {
    const soundFilaPeth = path.resolve(soundPath);
    let soundBuffer = await decoder.decode(fs.readFileSync(soundFilaPeth));
    soundBuffer = soundBuffer.channelData[chunnel];

    const shortTimeSounds = this.chunk(soundBuffer);
    shortTimeSounds.pop();

    return shortTimeSounds;
  }

  async separation() {
    for (let i = 0; i < config.numSamples; i++) {
      let p = path.join(__dirname, 'sounds', i + '.wav');
      let buffers = [await this.separateSound(p, 0), await this.separateSound(p, 1)];
      // random selection
      let index = 0;
      let randIndex = Array(buffers[0].length).fill(0);
      randIndex = randIndex.map((elem) => {
        return index++;
      });
      randIndex = _.shuffle(randIndex).slice(0, config.samplesPerSong);
      let randL = [];
      let randR = [];
      for (let i = 0; i < randIndex.length; i++) {
        randL.push(buffers[0][randIndex[i]]);
        randR.push(buffers[1][randIndex[i]]);
      }

      this.sounds.push({ buffers: [randL, randR] });

      console.log(i + ' / ' + config.numSamples);
    }
  }

  applyFilter() {
    for (let i = 0; i < this.sounds.length; i++) {
      const filteredL = this.sounds[i].buffers[0].map((buffer) => {
        let L = buffer;
        L = peaking.peaking(L, this.randomFilterParam());
        return L;
      });
      const filteredR = this.sounds[i].buffers[1].map((buffer) => {
        let R = buffer;
        R = peaking.peaking(R, this.randomFilterParam());
        return R;
      });

      this.sounds[i]['filteredBuffers'] = [filteredL, filteredR];
    }
  }

  makePair() {
    for (let sound of this.sounds) {
      const buffers = sound.buffers;
      const filteredBuffers = sound.filteredBuffers;

      for (let i = 0; i < buffers[0].length; i++) {
        let bufferFft = [
          this.fft(tf.tensor(buffers[0][i])),
          this.fft(tf.tensor(buffers[1][i]))
        ];
        let filteredFft = [
          this.fft(tf.tensor(filteredBuffers[0][i])),
          this.fft(tf.tensor(filteredBuffers[1][i]))
        ];
        let dataSet = [
          [
            [tf.real(bufferFft[0]).arraySync(), tf.imag(bufferFft[0]).arraySync()], // not filtered L-chunnel
            [tf.real(bufferFft[1]).arraySync(), tf.imag(bufferFft[1]).arraySync()]  // not filtered R-chunnel
          ],
          [
            [tf.real(filteredFft[0]).arraySync(), tf.imag(filteredFft[0]).arraySync()], // filtered L-chunnel
            [tf.real(filteredFft[1]).arraySync(), tf.imag(filteredFft[1]).arraySync()]  // filtered R-chunnel
          ]
        ];

        this.dataSets.push(dataSet);
      }
    }

    this.dataSets = _.shuffle(this.dataSets);
  }

  // log10(x) {
  //   return tf.tidy(() => {
  //     const numerator = tf.log(x);
  //     const denominator = tf.log(tf.scalar(10));
  //     return tf.div(numerator, denominator);
  //   });
  // }

  fft(input) {
    const fft = input.rfft();
    input.dispose();
    // console.log('fft: ', tf.memory());

    return fft;
  }

  ifft(input) {
    let ifft = input.irfft();
    let array = ifft.arraySync();

    input.dispose();
    ifft.dispose();
    // console.log('ifft: ', tf.memory());

    return array;
  }

  synth(buffer) {
    let output = [];

    // synthesis short-time buffer
    for (let i = 0; i < buffer.length; i++) {
      let hop = buffer[i].slice(0, buffer[i].length / 2);
      let overwrap = (i === 0) ?
          _.fill(Array(buffer[i].length / 2), 0)
        : buffer[i - 1].slice(buffer[i].length / 2, buffer[i - 1].length);

      for (let j = 0; j < hop.length; j++) {
        output.push(hop[j] + overwrap[j]);
      }
      overwrap = hop;
      hop = [];
    }

    return new Float32Array(output);
  }

  async decode(buffers) {
    let output = {
      sampleRate: 44100,
      float: true,
      channelData: buffers
    }
    return await encoder.encode(output);
  }

  nextBatch() {
    return tf.tidy(() => {
      let xbatchs = [];
      let ybatchs = [];
      for (let i = 0; i < config.bathSize; i++) {
        xbatchs.push(
          [
            [
              this.dataSets[this.batchIndex][0][0][0],  // L-chunnel real
              this.dataSets[this.batchIndex][0][0][1]   // L-chunnel image
            ],
            [
              this.dataSets[this.batchIndex][0][1][0],  // R-chunnel real
              this.dataSets[this.batchIndex][0][1][1]   // R-chunnel image
            ]
          ]
        );
        ybatchs.push(
          [
            [
              this.dataSets[this.batchIndex][1][0][0],  // L-chunnel real
              this.dataSets[this.batchIndex][1][0][1]   // L-chunnel image
            ],
            [
              this.dataSets[this.batchIndex][1][1][0],  // R-chunnel real
              this.dataSets[this.batchIndex][1][1][1]   // R-chunnel image
            ]
          ]
        );
        this.batchIndex++;
      }

      const x = tf.tensor(xbatchs, [config.bathSize, 2, 2, this.shortTimeSamples / 2 + 1]);
      const y = tf.tensor(ybatchs, [config.bathSize, 2, 2, this.shortTimeSamples / 2 + 1]);

      return { xs: x, ys: y };
    });
  }
}

module.exports = Data;
