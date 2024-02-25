import { pipeline, env } from "@xenova/transformers";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Due to a bug in onnxruntime-web, we must disable multithreading for now.
// See https://github.com/microsoft/onnxruntime/issues/14445 for more information.
// env.backends.onnx.wasm.numThreads = 1;

class PipelineSingleton {
  static task = "feature-extraction";
  static model = "Supabase/gte-small";
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}

let model = null;
let progress = null;

/**
 * Asynchronously loads the model.
 *
 * This function gets the pipeline instance which will load and build the model when run for the first time.
 * It also provides a way to track the progress of the pipeline creation, which can be used to update a UI element like a progress bar.
 *
 * @async
 * @function
 * @throws {Error} If the model cannot be loaded, an error is thrown with a message detailing the reason.
 *
 * @example
 * try {
 *   await loadModel();
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function loadModel() {
  try {
    // Get the pipeline instance. This will load and build the model when run for the first time.
    model = await PipelineSingleton.getInstance((data) => {
      // You can track the progress of the pipeline creation here.
      // e.g., you can send `data` back to the UI to indicate a progress bar
      // can be accessed via vagueFinder.getProgress()
      progress = data;
    });
  } catch (error) {
    throw new Error(`Unable to load Model due to ${error}`);
  }
}

/**
 * Throws an error indicating that the model has not been loaded.
 *
 *
 * @function
 * @throws {Error} Always throws an error indicating that the model has not been loaded.
 */

function modelNotLoadedErrorMessage() {
  throw new Error("Model has not been loaded, use vagueFinder.loadModel()");
}

/**
 * Asynchronously classifies the similarity between two sentences.
 *
 * This function takes two sentences and their respective embeddings and cache flags as input.
 * It calculates the embeddings for the sentences if they are not cached.
 * Then, it calculates the cosine similarity between the two embeddings.
 * It returns an object containing the two sentences, their similarity score, and the embedding of the first sentence.
 *
 * @async
 * @function
 * @param {string} sentenceOne - The first sentence to be compared.
 * @param {string} sentenceTwo - The second sentence to be compared.
 * @param {Array<number>} embedding1Cache - The cached embedding for the first sentence.
 * @param {boolean} doesCache1Exist - Flag indicating whether the embedding for the first sentence is cached.
 * @param {Array<number>} embedding2Cache - The cached embedding for the second sentence.
 * @param {boolean} doesCache2Exist - Flag indicating whether the embedding for the second sentence is cached.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<Object>} A Promise that resolves to an object containing the two sentences, their similarity score, and the embedding of the first sentence.
 *
 * @example
 * try {
 *   const result = await classify(sentence1, sentence2, embedding1Cache, true, embedding2Cache, false);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const classify = async (
  sentenceOne,
  sentenceTwo,
  embedding1Cache,
  doesCache1Exist,
  embedding2Cache,
  doesCache2Exist
) => {
  if (!doesCache2Exist && !model) {
    modelNotLoadedErrorMessage();
    return;
  }

  let tick = performance.now();
  let embedding1 = null;
  let embedding2 = null;

  if (doesCache1Exist) {
    embedding1 = embedding1Cache;
  } else {
    embedding1 = await model(sentenceOne, {
      pooling: "mean",
      normalize: true,
    });
  }

  if (doesCache2Exist) {
    embedding2 = embedding2Cache;
  } else {
    embedding2 = await model(sentenceTwo, {
      pooling: "mean",
      normalize: true,
    });
  }

  let tock = performance.now();
  console.log("time", tock - tick);
  if (!doesCache1Exist) {
    embedding1 = Array.from(embedding1.data);
  }
  if (!doesCache2Exist) {
    embedding2 = Array.from(embedding2.data);
  }

  const similarity = calculateCosineSimilarity(embedding1, embedding2);

  // console.log(similarity);

  let result = similarity;

  function calculateCosineSimilarity(embedding1, embedding2) {
    // Calculate dot product and magnitudes
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      magnitude1 += embedding1[i] * embedding1[i];
      magnitude2 += embedding2[i] * embedding2[i];
    }
    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    // Calculate cosine similarity
    const similarity = dotProduct / (magnitude1 * magnitude2);
    return similarity;
  }

  return {
    sentenceOne: sentenceOne,
    sentenceTwo: sentenceTwo,
    alike: result,
    embedding1Cache: embedding1,
  };
};

/**
 * Asynchronously compares a sentence to an array of sentences.
 *
 * This function takes a sentence and an array of sentences, and a cache flag as input.
 * It calculates the similarity between the input sentence and each sentence in the array.
 * It returns an object containing the input sentence and the array of sentences with their similarity scores.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to be compared.
 * @param {Array<string|Object>} array - The array of sentences to be compared. Each element can be a string or an object with `sentenceTwo` and `embedding` properties.
 * @param {boolean} doesCache2Exist - Flag indicating whether the embeddings for the sentences in the array are cached.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<Object>} A Promise that resolves to an object containing the input sentence and the array of sentences with their similarity scores.
 *
 * @example
 * try {
 *   const result = await compareSentenceToArray(sentence, array, true);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const compareSentenceToArray = async (sentence, array, doesCache2Exist=false) => {
  if (!doesCache2Exist && !model) {
    modelNotLoadedErrorMessage();
    return;
  }
  let cache = null;
  array = [...array]; //Creating a copy, so that we don't alter the original;
  for (let i = 0; i < array.length; i++) {
    const { sentenceTwo, alike, embedding1Cache } = await classify(
      sentence,
      array[i].sentenceTwo ? array[i].sentenceTwo : array[i],
      cache,
      i !== 0,
      array[i].embedding ? array[i].embedding : null,
      doesCache2Exist
    );
    if (i === 0) {
      cache = embedding1Cache;
    }
    array[i] = { sentenceTwo: sentenceTwo, alike: alike };
  }

  return {
    sentenceOne: sentence,
    array: array,
  };
};

/**
 * Asynchronously compares a sentence to an array of sentences and returns the results in order of similarity.
 *
 * This function takes a sentence and an array of sentences as input. It compares the input sentence to each sentence in the array using the `compareSentenceToArray` function, which calculates the cosine similarity between the sentences.
 * The result is an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity. The array is sorted in descending order of similarity.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to compare to the array of sentences.
 * @param {Array<string>} array - The array of sentences to compare to the input sentence.
 * @returns {Object} The result object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity, sorted in descending order of similarity.
 * @throws {Error} If the model has not been loaded.
 *
 * @example
 * try {
 *   const result = await arrayInOrder("This is a sentence.", ["This is another sentence.", "Yet another sentence."]);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const arrayInOrder = async (sentence, array) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  array = [...array]; //Creating a copy, so that we don't alter the original;
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    array,
    false
  );

  returnedArray.sort((a, b) => b.alike - a.alike);

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
};

/**
 * Returns the progress of the model loading process.
 *
 * If the model is loading, it returns an Object that represents the progress of the model loading process.
 *
 * @function
 * @returns {Object} The progress of the model loading process.
 *
 * @example
 * try {
 *   const progress = getProgress();
 *   console.log(progress);
 * } catch (error) {
 *   console.error(error);
 * }
 */

function getProgress() {
  return progress;
}

/**
 * Compares two sentences using the loaded model.
 *
 * This function takes two sentences as input and uses the `classify` function to compare them. If the model has not been loaded, it throws an error.
 *
 * @function
 * @param {string} sentenceOne - The first sentence to compare.
 * @param {string} sentenceTwo - The second sentence to compare.
 * @returns {Object} The result object containing the two input sentences and the calculated similarity.
 * @throws {Error} If the model has not been loaded.
 *
 * @example
 * try {
 *   const result = compareTwoSentences("This is a sentence.", "This is another sentence.");
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

function compareTwoSentences(sentenceOne, sentenceTwo) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }

  const { alike } = classify(
    sentenceOne,
    sentenceTwo,
    null,
    false,
    null,
    false
  );

  return { sentenceOne, sentenceTwo, alike };
}

/**
 * Asynchronously generates embeddings for an array of sentences.
 *
 * This function takes an array of sentences as input.
 * It generates embeddings for each sentence in the array using the model.
 * It returns an array of objects, each containing a sentence and its corresponding embedding.
 *
 * Note: This function creates a copy of the input array to avoid altering the original.
 *
 * @async
 * @function
 * @param {Array<string>} array - The array of sentences for which embeddings are to be generated.
 * @throws {Error} If the model is not loaded, an error is thrown.
 * @returns {Promise<Array<Object>>} A Promise that resolves to an array of objects, each containing a sentence and its corresponding embedding.
 *
 * @example
 * try {
 *   const result = await getCached(array);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function getCached(array) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  array = [...array]; //Creating a copy, so that we don't alter the original;
  let returnedArray = [];
  for (let i = 0; i < array.length; i++) {
    let embedding = await model(array[i], {
      pooling: "mean",
      normalize: true,
    });
    embedding = Array.from(embedding.data);
    returnedArray[i] = { sentenceTwo: array[i], embedding: embedding };
  }

  return returnedArray;
}

/**
 * Compares a sentence to an array of cached sentences.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to compare.
 * @param {Array} cachedArray - The array of cached sentences to compare against.
 * @returns {Promise<Object>} An object containing the original sentence and an array of comparison results.
 *
 * @example
 * const result = await cachedCompareSentenceToArray('Hello world', cachedSentences);
 * console.log(result);
 */

async function cachedCompareSentenceToArray(sentence, cachedArray) {
  cachedArray.map((item) => {
    if (!item.sentenceTwo) {
      throw new Error(
        "Each item in the cachedArray must have a sentenceTwo property"
      );
    }
    return {
      sentenceTwo: item.sentenceTwo,
      embedding: [...item.embedding],
    };
  });
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    cachedArray,
    true
  );

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
}

/**
 * Asynchronously sorts an array of sentences based on their similarity to a given sentence.
 *
 * This function takes a sentence and an array of sentences as input.
 * It calculates the similarity between the input sentence and each sentence in the array.
 * It then sorts the array based on the similarity scores in descending order.
 * It returns an object containing the input sentence and the sorted array of sentences with their similarity scores.
 *
 * This function differs from `arrayInOrder` in that it expects the array of sentences to already have cached embeddings.
 * This function is useful when you have a large array of sentences and you want to cache their embeddings to avoid recalculating them each time you compare a new sentence to the array.
 *
 * @async
 * @function
 * @param {string} sentence - The sentence to be compared.
 * @param {Array<string|Object>} cachedArray - The array of sentences to be compared. Each element is a object with `sentenceTwo` and `embedding` properties.
 * @returns {Promise<Object>} A Promise that resolves to an object containing the input sentence and the sorted array of sentences with their similarity scores.
 *
 * @example
 * try {
 *   const result = await cachedArrayInOrder(sentence, array);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

async function cachedArrayInOrder(sentence, cachedArray) {
  cachedArray.map((item) => {
    if (!item.sentenceTwo) {
      throw new Error(
        "Each item in the cachedArray must have a sentenceTwo property"
      );
    }
    return {
      sentenceTwo: item.sentenceTwo,
      embedding: [...item.embedding],
    };
  });
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    cachedArray,
    true
  );

  returnedArray.sort((a, b) => b.alike - a.alike);

  return {
    sentenceOne: sentenceOne,
    array: returnedArray,
  };
}

/**
 * The `vagueFinder` object provides a set of methods for comparing sentences using a loaded model.
 *
 * @namespace
 * @property {function} loadModel - Loads the model. See {@link loadModel}.
 * @property {function} getProgress - Returns the progress of the model loading process. See {@link getProgress}.
 * @property {function} compareTwoSentences - Compares two sentences using the loaded model. See {@link compareTwoSentences}.
 * @property {function} compareSentenceToArray - Compares a sentence to an array of sentences using the loaded model. See {@link compareSentenceToArray}.
 * @property {function} arrayInOrder - Compares a sentence to an array of sentences using the loaded model and returns the results in order of similarity. See {@link arrayInOrder}.
 * @property {function} getCached - Returns a cached array. See {@link getCached}.
 * @property {function} cachedCompareSentenceToArray - Compare a sentence to an array of cached sentences. See {@link cachedCompareSentenceToArray}.
 * @property {function} cachedArrayInOrder - Compares a sentence to an array of cached senteces and returns the results in order of similarity. See {@link cachedArrayInOrder}.
 */

const vagueFinder = {
  loadModel,
  getProgress,
  compareTwoSentences,
  compareSentenceToArray,
  arrayInOrder,
  getCached,
  cachedCompareSentenceToArray,
  cachedArrayInOrder,
};

export { vagueFinder };
