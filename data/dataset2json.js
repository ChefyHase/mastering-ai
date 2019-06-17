const Data = require('../data/data.js');
const fs = require('fs');
const config = require('../config.js');

(async () => {
  const data = new Data();
  await data.separation();
  data.applyFilter();
  data.makePair();

  for (let i = 0; i < config.trainEpoches; i++) {
    const filePath = config.dataSetPath + i + '.json';
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

    const buffer = data.dataSets.slice(i * config.batchSize, (i + 1) * config.batchSize);
    const json = JSON.stringify(buffer);
    const readStream = require('streamifier').createReadStream(Buffer.from(json));
    const writeStream = fs.createWriteStream(filePath);
    readStream.pipe(writeStream);
  }
})();
