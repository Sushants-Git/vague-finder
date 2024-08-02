# vague-finder

`vagueFinder` is a simple and easy to use package for sentences similarity operations. Was mainly created to be used in searching operations.

## Installation

Install `vagueFinder` using npm:

```bash
npm install vagueFinder
```

## Usage

### Example

```js
import { vagueFinder } from 'vague-finder';

await vagueFinder.loadModel();

const result = await vagueFinder.compareTwoSentences(
  "This is a sentence.",
  "This is another sentence."
);

console.log(
  `Similarity between "${result.sentenceOne}" and "${result.sentenceTwo}": ${result.alike}`
);
```

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
const result = await vagueFinder.compareTwoSentences(
  "This is a sentence.",
  "This is another sentence."
);

console.log(
  `Similarity between "${result.sentenceOne}" and "${result.sentenceTwo}": ${result.alike}`
);
```

### Comparing a Sentence to an Array of Sentences

To compare a sentence to an array of sentences, use the `compareSentenceToArray` method. This returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity.

```js
const result = await vagueFinder.compareSentenceToArray(
  "This is a sentence.",
  ["This is another sentence.", "Yet another sentence."],
  false,
);

result.array.forEach((comparison) => {
  console.log(
    `Similarity between "${result.sentenceOne}" and "${comparison.sentenceTwo}": ${comparison.alike}`
  );
});
```

### Comparing a Sentence to an Array of Sentences in Order of Similarity

To compare a sentence to an array of sentences and get the results in order of similarity, use the `arrayInOrder` method. This returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity.

```js
const result = await vagueFinder.arrayInOrder(
  "This is a sentence.",
  ["This is another sentence.", "Yet another sentence."]
);

result.array.forEach((comparison, index) => {
  console.log(
    `#${index + 1}: Similarity between "${result.sentenceOne}" and "${comparison.sentenceTwo}": ${comparison.alike}`
  );
});
```

### Caching Sentence Embeddings

To improve performance when comparing the same sentences multiple times, you can use the `getCached` method to cache the embeddings of an array of sentences. This method returns an array of objects, each containing a sentence from the input array and its corresponding embedding.

```js
const cachedSentences = await vagueFinder.getCached([
  "This is a sentence.",
  "This is another sentence.",
  "Yet another sentence.",
]);
console.log(cachedSentences);
```

Each object in the returned array has the following structure:

```js
{
  "sentenceTwo": "The sentence.",
  "embedding": [/* The embedding of the sentence. */]
}
```

### Comparing a Sentence to a Cached Array of Sentences

To compare a sentence to a cached array of sentences, use the `cachedCompareSentenceToArray` method. This method takes a sentence and an array of cached sentences (obtained from the `getCached` method) as input. It returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity.

```js
const cachedSentences = await vagueFinder.getCached([
  "This is a sentence.",
  "This is another sentence.",
  "Yet another sentence.",
]);

const result = await vagueFinder.cachedCompareSentenceToArray(
  "This is a sentence.",
  cachedSentences,
);

result.array.forEach((comparison) => {
  console.log(
    `Similarity between "${result.sentenceOne}" and "${comparison.sentenceTwo}": ${comparison.alike}`,
  );
});
```

### Comparing a Sentence to a Cached Array of Sentences and Sorting the Results

To compare a sentence to a cached array of sentences and get the results in order of similarity, use the `cachedArrayInOrder` method. This method takes a sentence and an array of cached sentences (obtained from the `getCached` method) as input. It returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity.

```js
const cachedSentences = await vagueFinder.getCached([
  "This is a sentence.",
  "This is another sentence.",
  "Yet another sentence.",
]);

const result = await vagueFinder.cachedArrayInOrder(
  "This is a sentence.",
  cachedSentences,
);

result.array.forEach((comparison) => {
  console.log(
    `Similarity between "${result.sentenceOne}" and "${comparison.sentenceTwo}": ${comparison.alike}`,
  );
});
```

### Getting Top Similar Sentences

To compare a sentence to an array of sentences and get the top similar sentences, use the `getTop` method. This method takes a sentence, an array of sentences, and the number of top results to return as input. It returns an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity.

```js
const sentences = [
  "This is a sentence.",
  "This is another sentence.",
  "Yet another sentence.",
];
const result = await vagueFinder.getTop("This is a sentence.", sentences, 2);
result.array.forEach((comparison) => {
  console.log(
    `The similarity between "${result.sentenceOne}" and "${comparison.sentenceTwo}" is ${comparison.alike}.`,
  );
});
```

## API

`loadModel()`

Asynchronously loads the model. This must be called before using any comparison methods. Throws an error if model loading fails.

`getProgress()`

Returns an object with information about the model loading progress.

`compareTwoSentences(sentenceOne, sentenceTwo)`

Compares two sentences using the loaded model. Returns an object containing both input sentences and their calculated similarity. Requires the model to be loaded first.

`compareSentenceToArray(sentence, array, doesCache2Exist)`

Compares a single sentence to an array of sentences. Returns an object with the input sentence and an array of comparison results, each containing a sentence from the input array and its similarity score. Requires a loaded model.

`arrayInOrder(sentence, array)`

Compares a sentence to an array of sentences and sorts results by similarity. Returns an object with the input sentence and an array of comparison results sorted in descending order of similarity. Requires a loaded model.

`getCached(array)`

Caches embeddings for an array of sentences. Returns an array of objects, each containing a sentence and its embedding. Requires a loaded model.

`cachedCompareSentenceToArray(sentence, cachedArray)`

Compares a sentence to a pre-cached array of sentences. Returns an object with the input sentence and an array of comparison results.

`cachedArrayInOrder(sentence, cachedArray)`

Compares a sentence to a pre-cached array of sentences and sorts results by similarity. Returns an object with the input sentence and a sorted array of comparison results. Requires a loaded model and valid cached array items.

`getTop(sentence, array, numberOfResults)`

Compares a sentence to an array of sentences and returns the top similar results. Returns an object with the input sentence and an array of the top comparison results, limited by `numberOfResults`. Requires a loaded model and a positive `numberOfResults` value.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

