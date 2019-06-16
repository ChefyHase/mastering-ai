const Data = require('./data/data.js');
const model = require('./model.js');
const tf = require('@tensorflow/tfjs-node-gpu');

(async ()=> {
  await model.train();
})();
