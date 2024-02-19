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
 * Asynchronously classifies the similarity between two sentences using cosine similarity.
 *
 * This function takes two sentences as input, converts them into embeddings using the model, and then calculates the cosine similarity between the embeddings.
 * The result is an object containing the two input sentences and the calculated similarity.
 *
 * @async
 * @function
 * @param {string} sentenceOne - The first sentence to compare.
 * @param {string} sentenceTwo - The second sentence to compare.
 * @returns {Object} The result object containing the two input sentences and the calculated similarity.
 * @throws {Error} If the model has not been loaded. 
*/

const classify = async (sentenceOne, sentenceTwo) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  let embedding1 = await model(sentenceOne, {
    pooling: "mean",
    normalize: true,
  });

  let embedding2 = await model(sentenceTwo, {
    pooling: "mean",
    normalize: true,
  });

  embedding1 = Array.from(embedding1.data);
  embedding2 = Array.from(embedding2.data);

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
  };
};

/**
 * Asynchronously compares a sentence to an array of sentences using cosine similarity.
 * 
 * This function takes a sentence and an array of sentences as input. It compares the input sentence to each sentence in the array using the `classify` function, which calculates the cosine similarity between the sentences.
 * The result is an object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity to the input sentence.
 * 
 * @async
 * @function
 * @param {string} sentence - The sentence to compare to the array of sentences.
 * @param {Array<string>} array - The array of sentences to compare to the input sentence.
 * @returns {Object} The result object containing the input sentence and an array of objects, each containing a sentence from the input array and the calculated similarity.
 * @throws {Error} If the model has not been loaded.
 * 
 * @example
 * try {
 *   const result = await compareSentenceToArray("This is a sentence.", ["This is another sentence.", "Yet another sentence."]);
 *   console.log(result);
 * } catch (error) {
 *   console.error(error);
 * }
 */

const compareSentenceToArray = async (sentence, array) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  array = [...array]; //Creating a copy, so that we don't alter the original;
  for (let i = 0; i < array.length; i++) {
    const { sentenceTwo, alike } = await classify(sentence, array[i]);
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
    array
  );

  returnedArray.sort((a, b) => {
    if (a.alike > b.alike) {
      return -1;
    } else if (a.alike < b.alike) {
      return 1;
    } else {
      return 0;
    }
  });

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
  return classify(sentenceOne, sentenceTwo);
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
 */

const vagueFinder = {
  loadModel: loadModel,
  getProgress: getProgress,
  compareTwoSentences: compareTwoSentences,
  compareSentenceToArray: compareSentenceToArray,
  arrayInOrder: arrayInOrder,
};

export { vagueFinder };
