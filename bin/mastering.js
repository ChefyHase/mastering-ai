const tf = require('@tensorflow/tfjs-node-gpu');
const encoder = require('wav-encoder');
const decoder = require('wav-decoder');
const fs = require('fs');
const path = require('path');

const Data = require('../data/data.js');

(async () => {
  const data = new Data();
  const model = await tf.loadLayersModel('file://mastering-ai/model.json');

  let bufferL = tf.tensor(await data.separateSound(__dirname + '/test.wav', 0));
  let bufferR = tf.tensor(await data.separateSound(__dirname + '/test.wav', 1));
  let xL = data.fft(bufferL);
  let xR = data.fft(bufferR);
  bufferL.dispose();
  bufferR.dispose();

  let xLRealTns = tf.real(xL);
  let xLReal = xLRealTns.arraySync();
  let xLImagTns = tf.imag(xL);
  let xLImag = xLImagTns.arraySync();

  let xRRealTns = tf.real(xR);
  let xRReal = xRRealTns.arraySync();
  let xRImagTns = tf.imag(xR);
  let xRImag = xRImagTns.arraySync();

  xLRealTns.dispose();
  xLImagTns.dispose();
  xRRealTns.dispose();
  xRImagTns.dispose();
  xL.dispose();
  xR.dispose();
  // console.log('Real/Imag Array: ', tf.memory());

  let xs = [];
  for (let i = 0; i < xLReal.length; i++) {
    let set = [
      [
        xLReal[i], xLImag[i]
      ],
      [
        xRReal[i], xRImag[i]
      ]
    ];
    xs.push(set);
  }
  xs = tf.tensor(xs, [xs.length, 2, 2, data.shortTimeSamples / 2 + 1]);
  // console.log('xs tensor: ', tf.memory());

  let ys = model.predictOnBatch(xs);
  xs.dispose();
  // console.log('ys tensor: ', tf.memory());

  ys = ys.arraySync();
  // console.log('ys to array: ', tf.memory());

  let yL = [];
  let yR = [];
  for (let y of ys) {
    let complex = tf.complex(y[0][0], y[0][1]);
    yL.push(data.ifft(complex)[0]);
    complex.dispose();
    complex = tf.complex(y[1][0], y[1][1]);
    yR.push(data.ifft(complex)[0]);
    complex.dispose();
  }
  const synth = [data.synth(yL), data.synth(yR)];
  const outputBuffer = await data.decode(synth);
  fs.writeFileSync(path.join(__dirname, 'output.wav'), Buffer.from(outputBuffer));
})();
