# vague-finder

`vagueFinder` is a simple and easy to use package for sentences similarity operations. Was mainly created to be used in searching operations.

## Installation

Install `vagueFinder` using npm:

```bash
npm install vagueFinder
```

## Usage

### Loading the Model

Before you can compare sentences, you need to load the model. This is an asynchronous operation. The model loaded is small and fast enough for most use cases.

```js
await vagueFinder.loadModel();
```

### Tracking Progress

You can get the progress of the model loading process. This returns an Object containing the progress information.

```js
const progress = vagueFinder.getProgress();
console.log(progress);
```

example output : 

```json
{
    "status": "progress",
    "name": "Supabase/gte-small",
    "file": "onnx/model_quantized.onnx",
    "progress": 1.5894844146421874,
    "loaded": 540654,
    "total": 34014426
}
```

### Comparing Two Sentences

To compare two sentences, use the `compareTwoSentences` method. This returns an object containing the two input sentences and the calculated similarity.

```js
const result = await vagueFinder.compareTwoSentences("This is a sentence.", "This is another sentence.");
console.log(`The similarity between "${result.sentenceOne}" and "${result.sentenceTwo}" is ${result.similarity}.`);
```

### Comparing a Sentence to an Array of Sentences

To compare a sentence to an array of sentences, use the `compareSentenceToArray` method. This returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity.

```js
const result = await vagueFinder.compareSentenceToArray("This is a sentence.", ["This is another sentence.", "Yet another sentence."]);
result.array.forEach((comparison) => {
  console.log(`The similarity between "${result.sentence}" and "${comparison.sentence}" is ${comparison.similarity}.`);
});
```

### Comparing a Sentence to an Array of Sentences in Order of Similarity

To compare a sentence to an array of sentences and get the results in order of similarity, use the `arrayInOrder` method. This returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity.

```js
const result = await vagueFinder.arrayInOrder("This is a sentence.", ["This is another sentence.", "Yet another sentence."]);
result.array.forEach((comparison, index) => {
  console.log(`#${index + 1}: The similarity between "${result.sentence}" and "${comparison.sentence}" is ${comparison.similarity}.`);
});
```


## API

`loadModel()`

Asynchronously loads the model. This must be called before any of the comparison methods. Throws an error if the model cannot be loaded.

`getProgress()`

You can get the progress of the model loading process. This returns an Object containing the progress information.

`compareTwoSentences(sentenceOne, sentenceTwo)`

Compares two sentences using the loaded model. Returns an object containing the two input sentences and the calculated similarity. Throws an error if the model has not been loaded.

`compareSentenceToArray(sentence, array)`

Compares a sentence to an array of sentences using the loaded model. Returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity. Throws an error if the model has not been loaded.

`arrayInOrder(sentence, array)`

Compares a sentence to an array of sentences using the loaded model and returns the results in order of similarity. Returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity. Throws an error if the model has not been loaded.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

