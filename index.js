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

async function loadModel() {
  // Get the pipeline instance. This will load and build the model when run for the first time.
  model = await PipelineSingleton.getInstance((data) => {
    // You can track the progress of the pipeline creation here.
    // e.g., you can send `data` back to the UI to indicate a progress bar
    progress = data;
    // console.log("progress", data);
  });
}

function modelNotLoadedErrorMessage() {
  console.log("Model has not been loaded, use vagueFinder.loadModel()");
}

const classify = async (sentenceOne, sentenceTwo) => {

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
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
    }

    // Calculate magnitudes
    let magnitude1 = 0;
    let magnitude2 = 0;
    for (let i = 0; i < embedding1.length; i++) {
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

const compareSentenceToArray = async (sentence, array) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  for (let i = 0; i < array.length; i++) {
    await classify(sentence, array[i]).then(({ sentenceTwo, alike }) => {
      array[i] = { senteceTwo: sentenceTwo, alike: alike };
    });
  }

  return {
    sentenceOne: sentence,
    array: array,
  };
};

const arrayInOrder = async (sentence, array) => {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  const { sentenceOne, array: returnedArray } = await compareSentenceToArray(
    sentence,
    array,
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

function getProgress() {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  return progress;
}

function compareTwoSentences(sentenceOne, sentenceTwo) {
  if (!model) {
    modelNotLoadedErrorMessage();
    return;
  }
  return classify(sentenceOne, sentenceTwo);
}

const vagueFinder = {
  loadModel: loadModel,
  getProgress: getProgress,
  compareTwoSentences: compareTwoSentences,
  compareSentenceToArray: compareSentenceToArray,
  arrayInOrder: arrayInOrder,
};

export { vagueFinder };
