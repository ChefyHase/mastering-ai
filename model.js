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

    const conv1d = tf.layers.conv2d({ dataFormat: 'channelsFirst', filters: 30, kernelSize: [1, 10], strides: 4 }).apply(input);
    const encoderActiv = tf.layers.leakyReLU().apply(conv1d);

    const flatten = tf.layers.flatten().apply(conv1d);
    const dense = tf.layers.dense({ units: 64 }).apply(flatten);
    const decoderActiv = tf.layers.leakyReLU().apply(dense);

    const decode1 = tf.layers.dense({ units: 512 }).apply(decoderActiv);
    const decoderActiv1 = tf.layers.leakyReLU().apply(decode1);

    const decodeDense = tf.layers.dense({ units: 4100 }).apply(decode1);
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
          batchSize: 100,
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
