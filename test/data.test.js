const Data = require('../data/data.js');

// test('separateSound', async () => {
//   const data = new Data();
//   const res = await data.separateSound();
// });
//
// test('applyFilter', async () => {
//   const data = new Data();
//   await data.separateSound();
//   data.applyFilter();
//
//   console.log(data.sounds[0].buffers[0]);
//   console.log(data.sounds[0].filteredBuffers[0]);
// });
//
// test('makePair', async () => {
//   const data = new Data();
//   await data.separateSound();
//   data.applyFilter();
//   data.makePair();
//
//   console.log(data.dataSets)
// });

test('nextBatch', async () => {
  const data = new Data();
  await data.separation();
  data.applyFilter();
  data.makePair();
  const tns = data.nextBatch();
});
