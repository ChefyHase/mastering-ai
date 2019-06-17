const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const input = tf.input({ shape: [2, 2, 1025] });

    const conv1d = tf.layers.conv2d({ dataFormat: 'channelsFirst', filters: 5, kernelSize: 2, strides: 1 }).apply(input);
    const encoderActiv = tf.layers.leakyReLU().apply(conv1d);

    const flatten = tf.layers.flatten().apply(conv1d);
    const dense = tf.layers.dense({ units: 512 }).apply(flatten);
    const decoderActiv = tf.layers.leakyReLU().apply(dense);

    const decode1 = tf.layers.dense({ units: 1024 }).apply(decoderActiv);
    const decoderActiv1 = tf.layers.leakyReLU().apply(decode1);
    const decode2 = tf.layers.dense({ units: 2048 }).apply(decoderActiv1);
    const decoderActiv2 = tf.layers.leakyReLU().apply(decode2);

    const decodeDense = tf.layers.dense({ units: 4100 }).apply(decoderActiv2);
    const decoderA = tf.layers.leakyReLU().apply(decodeDense);
    const reshape = tf.layers.reshape({ targetShape: [2, 2, 1025] }).apply(decoderA);

    const model = tf.model({ inputs: input, outputs: reshape });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < config.trainEpoches; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
          epochs: 30,
          shuffle: true,
          validationSplit: 0.3
      });

      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      await this.model.save('file://mastering-ai');

      xs.dispose();
      ys.dispose();

      // let buffer = await data.separateSound(__dirname + '/test.wav');
      // let xs = data.fft(tf.tensor(buffer));
    }
  }
}

module.exports = new Model();
