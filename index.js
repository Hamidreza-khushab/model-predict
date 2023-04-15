// // 1. ایجاد فایل CSV حاوی داده‌های تاریخی قیمت زوج ارزی EUR/USD
// // در این مثال، از داده‌های موجود در سایت Investing.com استفاده شده است.
// // داده‌ها در فایل eur_usd.csv ذخیره شده‌اند.

// // 2. خواندن داده‌های فایل CSV با استفاده از کتابخانه Papaparse
// const Papa = require('papaparse');
// const fs = require('fs');

// const csvFilePath = 'eur_usd.csv';
// const csvFile = fs.readFileSync(csvFilePath, 'utf8');

// const data = Papa.parse(csvFile, {
//   header: true,
//   dynamicTyping: true,
// }).data;

// // 3. ذخیره داده‌های خوانده شده از فایل CSV در آرایه‌ای چند بعدی
// // اینجا یک آرایه با سه بُعد باز و بسته شدن قیمت و حجم معاملات ساخته شده است.
// const tensorData = data.map(d => [d.Open, d.Close, d.Volume]).reverse();
// const xs = tensorData.slice(0, tensorData.length - 1);
// const ys = tensorData.slice(1);

// // 4. ایجاد شبکه عصبی با استفاده از کتابخانه TensorFlow.js
// const tf = require('@tensorflow/tfjs-node');

// const inputLayerShape = [null, 2]; // تعداد نمونه‌ها نامشخص است و بُعدهای ورودی دوتایی هستند (باز و بسته شدن قیمت)
// const model = tf.sequential();
// model.add(tf.layers.lstm({
// units: 50,
// inputShape: inputLayerShape,
// returnSequences: true,
// }));
// model.add(tf.layers.dropout({ rate: 0.2 }));
// model.add(tf.layers.lstm({ units: 50, returnSequences: false }));
// model.add(tf.layers.dropout({ rate: 0.2 }));
// model.add(tf.layers.dense({ units: 2 }));
// model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

// // 5. آموزش شبکه عصبی با استفاده از داده‌های تاریخی
// const batchSize = 32;
// const epochs = 50;
// const trainData = tf.data.zip({ xs: tf.data.array(xs), ys: tf.data.array(ys) })
// .shuffle(tensorData.length, 42)
// .batch(batchSize);
// await model.fitDataset(trainData, {
// epochs: epochs,
// callbacks: {
// onEpochEnd: async (epoch, logs) => {
// console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
// },
// },
// });

// // 6. بررسی عملکرد شبکه عصبی با استفاده از داده‌های تست
// // اینجا داده‌های تست از داده‌های تاریخی جدا شده و پیش‌بینی شبکه عصبی بر روی آن‌ها انجام می‌شود.
// const testData = tensorData.slice(0, 30).map(d => [d[0], d[1]]);
// const xTest = tf.tensor(testData);
// const predicted = model.predict(xTest);

// console.log('Predicted values:');
// predicted.print();
// console.log('True values:');
// console.log(tensorData.slice(1, 31).map(d => [d[0], d[1]]));

const Papa = require('papaparse');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

async function run() {
  const csvFilePath = 'eur_usd.csv';
  const csvFile = fs.readFileSync(csvFilePath, 'utf8');

  const data = Papa.parse(csvFile, {
    header: true,
    dynamicTyping: true,
  }).data;

  const tensorData = data.map(d => [d.Open, d.Close, d.Volume]).reverse();
  const xs = tensorData.slice(0, tensorData.length - 1);
  const ys = tensorData.slice(1);

  const inputLayerShape = [null, 2];
  const model = tf.sequential();
  model.add(tf.layers.lstm({
    units: 50,
    inputShape: inputLayerShape,
    returnSequences: true,
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.lstm({ units: 50, returnSequences: false }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 2 }));
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const batchSize = 32;
  const epochs = 50;
  const trainData = tf.data.zip({ xs: tf.data.array(xs), ys: tf.data.array(ys) })
    .shuffle(tensorData.length, 42)
    .batch(batchSize);
  await model.fitDataset(trainData, {
    epochs: epochs,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      },
    },
  });

  const testData = tensorData.slice(0, 30).map(d => [d[0], d[1]]);
  const xTest = tf.tensor(testData);
  const predicted = model.predict(xTest);

  console.log('Predicted values:');
  predicted.print();
  console.log('True values:');
  console.log(tensorData.slice(1, 31).map(d => [d[0], d[1]]));
}

run();
