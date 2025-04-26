/***********************************************************
 * CONFIGURATION
 ***********************************************************/

// Question metadata.
const QUESTIONS = [
  {
    key: "bite_order",
    text: "What bite order would you like?",
    contentDir: "bite_ordering"
  },
  {
    key: "ready_signal",
    text: "What ready signal would you like the robot to use?",
    contentDir: "ready_signal"
  },
  {
    key: "verbal",
    text: "Would you like the robot to be verbal during this meal?",
    contentDir: "be_verbal"
  },
  {
    key: "look_forward",
    text: "Would you typically be looking forward during this meal?",
    contentDir: "occlusion"
  },
  {
    key: "block_forward",
    text: "Is the robot uncomfortably blocking your forward sight?",
    contentDir: "occlusion"
  },
  {
    key: "look_left",
    text: "Would you typically be looking a little towards the left during this meal?",
    contentDir: "occlusion"
  },
  {
    key: "block_left",
    text: "Is the robot uncomfortably blocking your leftward sight?",
    contentDir: "occlusion"
  },
];

// Fallback option lists for the very first meal.
// These will be replaced automatically per-meal using prediction files.
const INITIAL_OPTIONS = {
  bite_order: ["Loading..."],
  ready_signal: ["Loading..."],
  verbal: ["Loading..."],
  look_forward: ["Yes", "No"],
  block_forward: ["Yes", "No"],
  look_left: ["Yes", "No"],
  block_left: ["Yes", "No"],
};

/***********************************************************
 * STATE
 ***********************************************************/
const state = {
  answers: [], // Array of per‐meal answer objects
  metadata: {}, // Stores loaded metadata per question
  predictions: {}, // Stores loaded predictions per question
  currentMeal: null, // Stores the current meal metadata
};

/***********************************************************
 * DOM ELEMENTS
 ***********************************************************/
const homeScreen = document.getElementById("home-screen");
const mealScreen = document.getElementById("meal-screen");
const thanksScreen = document.getElementById("thanks-screen");
const startBtn = document.getElementById("start-btn");
const nextBtn = document.getElementById("next-btn");
const mealTitle = document.getElementById("meal-title");
const mealImg = document.getElementById("meal-img");
const mealDesc = document.getElementById("meal-desc");
const formEl = document.getElementById("questions-form");

/***********************************************************
 * UTILS
 ***********************************************************/
function show(el) {
  el.classList.remove("hidden");
}

function hide(el) {
  el.classList.add("hidden");
}

/**
 * Generates a test URL for a specific series of answers
 * @param {Array} answers - Array of answer objects
 * @returns {string} The URL with encoded answers
 */
function generateTestUrl(answers) {
  const encodedAnswers = encodeURIComponent(JSON.stringify(answers));
  return `meal.html?answers=${encodedAnswers}`;
}

/**
 * Builds the content directory path for a given question based on previous answers.
 * Example: If user chose "right" for bite_order,
 * the path for bite_order would be: "content/bite_ordering/right/"
 * 
 * @param {string} questionKey - The key of the current question (e.g., "bite_order")
 * @returns {string} The full path to the content directory for this question
 */
function getContentPath(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;

  // Start with the base content directory and question type
  const pathParts = ['content', question.contentDir];

  // Add previous answers to build the path
  // For example, if we're on bite_order and the user chose specific options
  // for previous questions, we need to include those in the path
  if (question.contentDir !== 'occlusion') {
    for (let i = 0; i < state.answers.length; i++) {
      const answer = state.answers[i];
      pathParts.push(answer[questionKey].value);
    }
  }

  // Special handling for occlusion directory naming
  if (question.contentDir === 'occlusion') {
    // Build path by considering each answer's occlusion state in sequence
    for (const answer of state.answers) {
      const relevantPois = [];
      const occludedPois = [];
      
      // Check each occlusion-related answer for this specific answer
      if (answer.look_forward?.value === 'Yes') relevantPois.push('front');
      if (answer.look_left?.value === 'Yes') relevantPois.push('left');
      if (answer.block_forward?.value === 'Yes') occludedPois.push('front');
      if (answer.block_left?.value === 'Yes') occludedPois.push('left');
      
      // Create the directory name for this answer
      const relevantStr = relevantPois.length > 0 ? relevantPois.join('-') : 'none';
      const occludedStr = occludedPois.length > 0 ? occludedPois.join('-') : 'none';
      const occlusionDir = `${relevantStr}___${occludedStr}`;
      
      // Append this answer's occlusion directory to the path
      pathParts.push(occlusionDir);
    }
  }

  console.log("Content path for", questionKey, ":", pathParts.join('/'));
  return pathParts.join('/');
}

/***********************************************************
 * STATE MANAGEMENT
 ***********************************************************/
function compressState(answers) {
  // Convert answers to a minimal format
  const minimalAnswers = answers.map(answer => {
    const minimal = {};
    for (const [key, value] of Object.entries(answer)) {
      if (key === 'occlusion') {
        // Special handling for occlusion data
        minimal[key] = {
          r: value.relevant_pois,
          o: value.occluded_pois
        };
      } else {
        // For other answers, just store the value
        minimal[key] = value.value;
      }
    }
    return minimal;
  });
  
  // Convert to base64
  const json = JSON.stringify(minimalAnswers);
  return btoa(json);
}

function decompressState(compressed) {
  try {
    // Decode base64
    const json = atob(compressed);
    const minimalAnswers = JSON.parse(json);
    
    // Convert back to full format
    return minimalAnswers.map(answer => {
      const full = {};
      for (const [key, value] of Object.entries(answer)) {
        if (key === 'occlusion') {
          // Special handling for occlusion data
          full[key] = {
            relevant_pois: value.r,
            occluded_pois: value.o
          };
        } else {
          // For other answers, restore the full format
          full[key] = {
            value: value,
            metadata: null
          };
        }
      }
      return full;
    });
  } catch (error) {
    console.error('Error decompressing state:', error);
    return [];
  }
}

function getStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const compressed = params.get('state');
  const answers = compressed ? decompressState(compressed) : [];
  return { answers };
}

function updateUrlState() {
  const params = new URLSearchParams();
  const compressed = compressState(state.answers);
  params.set('state', compressed);
  
  // Update URL without reloading the page
  window.history.replaceState(
    {},
    '',
    `${window.location.pathname}?${params.toString()}`
  );
}

function navigateToMeal() {
  const params = new URLSearchParams();
  if (state.answers.length > 0) {
    const compressed = compressState(state.answers);
    params.set('state', compressed);
  }
  window.location.href = `meal.html?${params.toString()}`;
}

function navigateToThanks() {
  window.location.href = 'thanks.html';
}

/***********************************************************
 * CONTENT LOADING
 ***********************************************************/
async function loadMetadata(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  try {
    const contentPath = getContentPath(questionKey);
    
    const response = await fetch(`${contentPath}/metadata.json`);
    if (!response.ok) {
      console.error(`HTTP error loading metadata for ${questionKey}: ${response.status} ${response.statusText}`);
      return null;
    }
    
    const text = await response.text();
    if (!text.trim()) {
      console.error(`Empty metadata file for ${questionKey}`);
      return null;
    }
    
    try {
      return JSON.parse(text);
    } catch (parseError) {
      console.error(`Invalid JSON in metadata for ${questionKey}:`, text);
      return null;
    }
  } catch (error) {
    console.error(`Error loading metadata for ${questionKey}:`, error);
    return null;
  }
}

async function loadPredictions(questionKey) {
  const question = QUESTIONS.find(q => q.key === questionKey);
  if (!question) return null;
  
  // For the first meal, use the first option instead of predictions
  if (state.answers.length === 0) {
    // For verbal and occlusion questions, return appropriate defaults
    if (questionKey === 'verbal') {
      return 'True';  // Default to "Yes" for verbal
    }
    
    // Create a single occlusion data structure for the first meal and reuse it
    if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
      // Initialize with default occlusion data
      const occlusionData = {
        relevant_pois: [],  // Default to no for everything
        occluded_pois: [],
      };
      return occlusionData;
    }
  
    // For other questions, use the first option from metadata if available
    const metadata = state.metadata[questionKey];
    if (metadata && metadata.choices && metadata.choices.length > 0) {
      return metadata.choices[0];
    }
    
    // Fallback to initial options
    return INITIAL_OPTIONS[questionKey][0];
  }
  
  // For subsequent meals, use the normal prediction loading logic
  try {
    const contentPath = getContentPath(questionKey);

    // Special handling for occlusion questions
    if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
      // For occlusion questions, we should use the standard path      
      const response = await fetch(`${contentPath}/prediction.json`);
      const data = await response.json();
      return data;
    }

    const response = await fetch(`${contentPath}/prediction.txt`).then(res => res.text());
    return response;
  
  } catch (error) {
    console.error(`Error loading predictions for ${questionKey}:`, error);
    return null;
  }
}

async function loadCurrentMealInfo() {
  try {
    // Use getContentPath to get the correct path for occlusion questions
    const occlusionPath = getContentPath('look_forward');
    
    const response = await fetch(`${occlusionPath}/metadata.json`);
    const metadata = await response.json();
    
    // Create a descriptive meal context
    const foodItems = metadata.food_items.join(' and ');
    const dips = metadata.dips.join(' and ');
    const context = metadata.context.replace('_', ' ');
    const tableType = metadata.table_type.replace('_', ' ');
    
    const description = `Imagine you are having a meal in a <span class="context">${context}</span> setting at a <span class="table-type">${tableType}</span>. 
    On your plate, you have <span class="food-items">${foodItems}</span>${dips ? ` with <span class="dips">${dips}</span> for dipping` : ''}. 
    Please answer the following questions about this meal scenario.`;
    
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: `${occlusionPath}/bite_occlusion_image.png`,
      description: description
    };
    
  } catch (error) {
    console.error('Error loading meal info:', error);
    
    // Fallback to default values if loading fails
    state.currentMeal = {
      title: `Meal ${state.answers.length + 1} of 5`,
      image: 'content/occlusion/bite_occlusion_image.png',
      description: "Please answer the following questions about this meal scenario."
    };
  }
}

/***********************************************************
 * UI RENDERING
 ***********************************************************/
async function loadCurrentContent() {
  state.metadata = {};
  state.predictions = {};
  
  // Load meal info first
  await loadCurrentMealInfo();
  
  // Load metadata and predictions for each question
  for (const question of QUESTIONS) {
    const key = question.key;
    state.metadata[key] = await loadMetadata(key);
    state.predictions[key] = await loadPredictions(key);
  }
}

async function getOptionsForQuestion(questionKey) {
  // For verbal and occlusion questions, always use Yes/No
  if (questionKey === 'verbal') {
    return ['Yes', 'No'];
  }

  if (questionKey.startsWith('look_') || questionKey.startsWith('block_')) {
    return ['No', 'Yes'];
  }
  
  // For all other questions, use metadata choices if available
  const metadata = state.metadata[questionKey];
  if (metadata && metadata.choices) {
    return metadata.choices;
  }
  
  // Fall back to initial options if no metadata
  return INITIAL_OPTIONS[questionKey];
}

async function renderForm() {
  const form = document.getElementById("questions-form");
  form.innerHTML = ""; // Clear existing questions
  
  // Add preference rating at the beginning of the form
  await addPreferenceRating(form);
  
  // Add section heading for meal questions
  const sectionHeading = document.createElement("h3");
  sectionHeading.textContent = state.answers.length === 0 
    ? "Help the robot learn your preferences" 
    : "Fine-tune your personalized experience";
  sectionHeading.style.marginTop = "0.5rem";
  sectionHeading.style.marginBottom = "0.25rem";
  form.appendChild(sectionHeading);
  
  // Add section description
  const sectionDesc = document.createElement("p");
  sectionDesc.innerHTML = state.answers.length === 0
    ? "For your first meal, please select your preferences for each setting:"
    : "Please review the robot's personalized choices and adjust if needed:";
  sectionDesc.style.marginBottom = "0.5rem";
  form.appendChild(sectionDesc);
  
  for (const question of QUESTIONS) {
    const options = await getOptionsForQuestion(question.key);
    
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "0.5rem";
    wrapper.style.padding = "0.75rem";
    wrapper.style.border = "1px solid #e0e0e0";
    wrapper.style.borderRadius = "4px";
    wrapper.style.backgroundColor = "#fafafa";
    
    const label = document.createElement("label");
    label.setAttribute("for", question.key);
    label.className = 'meal-context';
    label.style.fontWeight = "500";
    label.style.display = "block";
    label.style.marginBottom = "0.75rem";
    
    // Get the prediction for this question
    let prediction = state.predictions[question.key];
    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }
    
    // Handle Yes/No predictions for verbal and occlusion questions
    if (question.key === 'verbal' && prediction) {
      prediction = prediction.trim() === 'True' ? 'Yes' : 'No';
    }
    
    // Handle occlusion predictions
    if ((question.key.startsWith('look_') || question.key.startsWith('block_')) && prediction) {
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      if (prediction.relevant_pois && prediction.occluded_pois) {
        // Use the prediction data directly from this question
        if (isLooking) {
          prediction = prediction.relevant_pois.includes(direction) ? 'Yes' : 'No';
        } else {
          prediction = prediction.occluded_pois.includes(direction) ? 'Yes' : 'No';
        }
      } else {
        // Fallback defaults if prediction data is missing
        if (state.answers.length === 0) {
          if (isLooking) {
            // Default to looking forward only
            prediction = direction === 'front' ? 'Yes' : 'No';
          } else {
            // Default to no occlusion
            prediction = 'No';
          }
        } else {
          prediction = isLooking && direction === 'front' ? 'Yes' : 'No';
        }
      }
    }
    
    // Create the question text based on the type
    if (question.key === 'bite_order') {
      if (state.answers.length === 0) {
        label.innerHTML = `The robot's initial selection is to serve your food as follows: <span class="context">${prediction || 'Loading...'}</span>. Is this what you prefer?`;
      } else {
        label.innerHTML = `The robot has learned to serve your food as follows: <span class="context">${prediction || 'Loading...'}</span>. Is this correct?`;
      }
    } else if (question.key === 'ready_signal') {
      if (state.answers.length === 0) {
        label.innerHTML = `The robot's initial selection is to use <span class="context">${prediction || 'a button'}</span> as a ready signal. Is this what you prefer?`;
      } else {
        label.innerHTML = `The robot has learned to use <span class="context">${prediction || 'a button'}</span> as a ready signal. Is this correct?`;
      }
    } else if (question.key === 'verbal') {
      if (state.answers.length === 0) {
        label.innerHTML = `Would you like the robot to be <span class="context">verbal</span> during this meal?`;
      } else {
        label.innerHTML = `The robot has learned that you prefer it to <span class="context">${prediction === 'Yes' ? 'be verbal' : 'remain silent'}</span> during meals. Is this correct?`;
      }
    } else if (question.key.startsWith('look_')) {
      const direction = question.key.includes('forward') ? 'forward' : 'left';
      if (state.answers.length === 0) {
        label.innerHTML = `Would you typically be looking <span class="context">${direction}</span> during this meal?`;
      } else {
        label.innerHTML = `The robot has learned that you typically ${prediction === 'Yes' ? 'look' : 'do not look'} <span class="context">${direction}</span> during meals. Is this correct?`;
      }
    } else if (question.key.startsWith('block_')) {
      const direction = question.key.includes('forward') ? 'forward' : 'left';
      if (state.answers.length === 0) {
        label.innerHTML = `Is the robot uncomfortably blocking your <span class="context">${direction}</span> sight?`;
      } else {
        label.innerHTML = `The robot has learned that it ${prediction === 'Yes' ? 'does' : 'does not'} block your <span class="context">${direction}</span> sight. Is this correct?`;
      }
    } else {
      label.textContent = question.text;
    }
    
    const select = document.createElement("select");
    select.id = question.key;
    select.name = question.key;
    select.style.width = "100%";
    select.style.padding = "0.5rem";
    select.style.fontSize = "1rem";
    select.style.borderRadius = "4px";
    select.style.border = "1px solid #ccc";
    
    // Add empty default option
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Select an option...";
    select.appendChild(defaultOption);
    
    // Add all available options
    if (Array.isArray(options)) {
      options.forEach((optionText) => {
        if (optionText) {  // Only add non-empty options
          const option = document.createElement("option");
          option.value = optionText;
          option.textContent = optionText;
          select.appendChild(option);
        }
      });
    }
    
    // If we have metadata for this question, add it as a data attribute
    const metadata = state.metadata[question.key];
    if (metadata) {
      select.dataset.metadata = JSON.stringify(metadata);
    }
    
    // Pre-select the predicted option if available
    if (prediction) {
      select.value = prediction;
    }
    
    select.addEventListener("change", checkFormCompletion);
    
    wrapper.appendChild(label);
    wrapper.appendChild(select);
    form.appendChild(wrapper);
  }
}

async function addPreferenceRating(form) {
  console.log("Adding preference rating to form");
  
  // Add preference rating question
  const ratingDiv = document.createElement("div");
  ratingDiv.className = "preference-rating";
  ratingDiv.style.marginBottom = "2.5rem";
  ratingDiv.style.padding = "1.25rem";
  ratingDiv.style.border = "1px solid #d0d0d0";
  ratingDiv.style.borderRadius = "6px";
  ratingDiv.style.backgroundColor = "#f5f8ff";
  ratingDiv.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
  
  // Create title for the rating section
  const ratingTitle = document.createElement("h3");
  ratingTitle.style.marginTop = "0";
  ratingTitle.style.marginBottom = "0.75rem";
  ratingTitle.style.fontSize = "1.25rem";
  
  let ratingText = "";
  ratingTitle.textContent = "Evaluate Personalization";
  ratingText = "To what extent do you prefer the default or the personalized choices overall? (1 = default is best, 4 = neutral, 7 = personalized is best)";
  
  ratingDiv.appendChild(ratingTitle);
  
  const ratingLabel = document.createElement("p");
  ratingLabel.htmlFor = "preference_rating";
  ratingLabel.innerHTML = ratingText;
  ratingLabel.className = "meal-context";
  ratingLabel.style.fontSize = "1.05rem";
  ratingLabel.style.marginBottom = "1rem";
  
  const ratingScale = document.createElement("div");
  ratingScale.style.display = "flex";
  ratingScale.style.alignItems = "center";
  ratingScale.style.marginBottom = "1rem";
  
  const ratingDesc1 = document.createElement("div");
  ratingDesc1.style.flex = "1";
  ratingDesc1.style.textAlign = "left";
  ratingDesc1.style.fontSize = "0.9rem";
  ratingDesc1.style.color = "#555";
  ratingDesc1.innerHTML = state.answers.length === 0 ? "" : "<strong>1</strong>: Not at all personalized";
  
  const ratingDesc2 = document.createElement("div");
  ratingDesc2.style.flex = "1";
  ratingDesc2.style.textAlign = "center";
  ratingDesc2.style.fontSize = "0.9rem";
  ratingDesc2.style.color = "#555";
  ratingDesc2.innerHTML = state.answers.length === 0 ? "" : "<strong>4</strong>: Moderately personalized";
  
  const ratingDesc3 = document.createElement("div");
  ratingDesc3.style.flex = "1";
  ratingDesc3.style.textAlign = "right";
  ratingDesc3.style.fontSize = "0.9rem";
  ratingDesc3.style.color = "#555";
  ratingDesc3.innerHTML = state.answers.length === 0 ? "" : "<strong>7</strong>: Perfectly personalized";
  
  ratingScale.appendChild(ratingDesc1);
  ratingScale.appendChild(ratingDesc2);
  ratingScale.appendChild(ratingDesc3);
  
  const ratingSelect = document.createElement("select");
  ratingSelect.id = "preference_rating";
  ratingSelect.name = "preference_rating";
  ratingSelect.style.width = "100%";
  ratingSelect.style.padding = "0.75rem";
  ratingSelect.style.marginTop = "0.5rem";
  ratingSelect.style.fontSize = "1.1rem";
  ratingSelect.style.border = "1px solid #ccc";
  ratingSelect.style.borderRadius = "4px";
  ratingSelect.style.backgroundColor = "#fff";

  // Add rating options 1-7
  for (let i = 1; i <= 7; i++) {
    const option = document.createElement("option");
    option.value = i.toString();
    option.textContent = i.toString();
    ratingSelect.appendChild(option);
  }
    
  // Pre-select the middle value (4 = moderate) for convenience
  ratingSelect.value = "4";
  
  ratingSelect.addEventListener("change", checkFormCompletion);
  
  ratingDiv.appendChild(ratingLabel);
  if (state.answers.length > 0) {
    ratingDiv.appendChild(ratingScale);
  }
  ratingDiv.appendChild(ratingSelect);
  
  // Add to the form
  form.appendChild(ratingDiv);
  
  // Debug log
  console.log("Preference rating added, form now has", form.querySelectorAll("select").length, "select elements");
}

async function renderPredictionSummary() {
  console.log("Rendering prediction summary");
  // Create summary section
  const form = document.getElementById("questions-form");
  const summarySection = document.createElement("div");
  summarySection.className = "prediction-summary";
  summarySection.style.marginBottom = "2rem";
  
  const title = document.createElement("h3");
  title.textContent = "Robot Decision Summary";
  title.style.marginBottom = "0.5rem";
  summarySection.appendChild(title);
  
  const description = document.createElement("p");
  // Change description based on whether this is the first meal
  if (state.answers.length === 0) {
    description.innerHTML = "The robot is not yet personalized. Here are the selections it would make for your first meal, where default options are the same as the \"personalized\" ones:";
  } else {
    description.innerHTML = "Based on your previous preferences, here's how the robot has personalized its choices for you:";
  }
  description.style.marginBottom = "1rem";
  summarySection.appendChild(description);
  
  // Create table for predictions
  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";
  
  // Add table header
  const thead = document.createElement("thead");
  thead.style.backgroundColor = "#f0f0f0";
  const headerRow = document.createElement("tr");
  
  const questionHeader = document.createElement("th");
  questionHeader.textContent = "Setting";
  questionHeader.style.padding = "0.75rem";
  questionHeader.style.textAlign = "left";
  headerRow.appendChild(questionHeader);
  
  const defaultHeader = document.createElement("th");
  defaultHeader.textContent = "Default";
  defaultHeader.style.padding = "0.75rem";
  defaultHeader.style.textAlign = "left";
  headerRow.appendChild(defaultHeader);
  
  const personalizedHeader = document.createElement("th");
  // Change column header for first meal
  personalizedHeader.textContent = "Personalized";
  personalizedHeader.style.padding = "0.75rem";
  personalizedHeader.style.textAlign = "left";
  headerRow.appendChild(personalizedHeader);
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Add table body
  const tbody = document.createElement("tbody");
  
  for (const question of QUESTIONS) {
    const options = await getOptionsForQuestion(question.key);
    // Get the default option (first in the list)
    const defaultOption = options.length > 0 ? options[0] : "None";
    
    // Get the personalized prediction
    let prediction = state.predictions[question.key];
    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }
    
    // Format prediction for display
    let formattedPrediction = prediction;
    
    // Handle Yes/No predictions for verbal and occlusion questions
    if (question.key === 'verbal' && prediction) {
      formattedPrediction = prediction.trim() === 'True' ? 'Yes' : 'No';
    }
    
    // Handle occlusion predictions
    if ((question.key.startsWith('look_') || question.key.startsWith('block_')) && prediction) {
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      if (prediction.relevant_pois && prediction.occluded_pois) {
        // Use the prediction data directly from this question
        if (isLooking) {
          formattedPrediction = prediction.relevant_pois.includes(direction) ? 'Yes' : 'No';
        } else {
          formattedPrediction = prediction.occluded_pois.includes(direction) ? 'Yes' : 'No';
        }
      } else {
        // Fallback defaults if prediction data is missing
        if (state.answers.length === 0) {
          if (isLooking) {
            // Default to looking forward only
            formattedPrediction = direction === 'front' ? 'Yes' : 'No';
          } else {
            // Default to no occlusion
            formattedPrediction = 'No';
          }
        } else {
          formattedPrediction = isLooking && direction === 'front' ? 'Yes' : 'No';
        }
      }
    }
    
    const row = document.createElement("tr");
    row.style.borderBottom = "1px solid #eee";
    
    const questionCell = document.createElement("td");
    questionCell.textContent = question.text;
    questionCell.style.padding = "0.75rem";
    row.appendChild(questionCell);
    
    const defaultCell = document.createElement("td");
    defaultCell.textContent = defaultOption;
    defaultCell.style.padding = "0.75rem";
    row.appendChild(defaultCell);
    
    const predictionCell = document.createElement("td");
    predictionCell.textContent = formattedPrediction || "No prediction";
    predictionCell.style.padding = "0.75rem";
    if (formattedPrediction !== defaultOption && formattedPrediction) {
      predictionCell.className = "personalized";
      predictionCell.style.fontWeight = "bold";
      predictionCell.style.color = "#1a73e8";
    }
    row.appendChild(predictionCell);
    
    tbody.appendChild(row);
  }
  
  table.appendChild(tbody);
  summarySection.appendChild(table);
  
  // Insert summary section at the beginning of the form
  form.insertBefore(summarySection, form.firstChild);
}

async function showMeal() {
  // Load content before showing the meal
  await loadCurrentContent();
  
  const mealTitle = document.getElementById("meal-title");
  const mealImg = document.getElementById("meal-img");
  
  mealTitle.textContent = state.currentMeal.title;
  mealImg.src = state.currentMeal.image;
  
  // Create a styled container for the meal description
  const descContainer = document.createElement("div");
  descContainer.style.backgroundColor = "#ffffff";
  descContainer.style.border = "1px solid #e0e0e0";
  descContainer.style.borderRadius = "6px";
  descContainer.style.padding = "1.25rem";
  descContainer.style.marginTop = "1rem";
  descContainer.style.marginBottom = "1.5rem";
  descContainer.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
  
  // Add heading for the scenario
  const scenarioHeading = document.createElement("h3");
  scenarioHeading.textContent = "Meal Scenario";
  scenarioHeading.style.marginTop = "0";
  scenarioHeading.style.marginBottom = "0.75rem";
  scenarioHeading.style.color = "#333";
  descContainer.appendChild(scenarioHeading);
  
  // Create the meal description element
  const mealDesc = document.createElement("p");
  mealDesc.id = "meal-desc";
  mealDesc.innerHTML = state.currentMeal.description;
  mealDesc.className = 'meal-context';
  mealDesc.style.margin = "0";
  mealDesc.style.lineHeight = "1.5";
  mealDesc.style.fontSize = "1.05rem";
  
  // Add the description to the container
  descContainer.appendChild(mealDesc);
  
  // Insert the container after the image
  const mealSection = document.querySelector('section');
  mealImg.parentNode.insertBefore(descContainer, mealImg.nextSibling);
  
  await renderForm();
  await renderPredictionSummary();
}

function checkFormCompletion() {
  const form = document.getElementById("questions-form");
  const nextBtn = document.getElementById("next-btn");
  const selects = form.querySelectorAll("select");
  
  // Check if all selects have a value
  let allAnswered = true;
  
  // Log each select element and its value for debugging
  selects.forEach(select => {
    console.log(`Select ${select.name}: value = "${select.value}"`);
    if (!select.value) {
      allAnswered = false;
      console.log(`Missing value for ${select.name}`);
    }
  });
  
  console.log("All questions answered:", allAnswered);
  nextBtn.disabled = !allAnswered;
}

async function collectAnswers() {
  const form = document.getElementById("questions-form");
  const answers = {};
  
  // Get the preference rating if it exists
  const preferenceRating = form.querySelector('select[name="preference_rating"]');
  if (preferenceRating && preferenceRating.value) {
    answers.preference_rating = {
      value: preferenceRating.value,
      metadata: null
    };
  }
  
  for (const question of QUESTIONS) {
    const select = form.querySelector(`select[name="${question.key}"]`);
    const value = select.value;
    
    // Handle special cases for storing answers
    if (question.key === 'verbal') {
      // Convert Yes/No back to True/False
      answers[question.key] = {
        value: value === 'Yes' ? 'True' : 'False',
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } 
    
    else if (question.key.startsWith('look_') || question.key.startsWith('block_')) {
      // For occlusion questions, we need to update the prediction.json structure
      const direction = question.key.includes('forward') ? 'front' : 'left';
      const isLooking = question.key.startsWith('look_');
      
      // Get or create the occlusion data
      if (!answers.occlusion) {
        answers.occlusion = {
          relevant_pois: [],
          occluded_pois: []
        };
      }
      
      // Update the appropriate list based on the answer
      if (value === 'Yes') {
        if (isLooking) {
          answers.occlusion.relevant_pois.push(direction);
        } else {
          answers.occlusion.occluded_pois.push(direction);
        }
      }
      
      // Store the individual answer as well
      answers[question.key] = {
        value,
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    } else {
      // For all other questions, store the value as is
      answers[question.key] = {
        value,
        metadata: select.dataset.metadata ? JSON.parse(select.dataset.metadata) : null
      };
    }
  }
  
  return answers;
}

async function handleNextClick() {
  // Collect and save answers
  const answers = await collectAnswers();
  state.answers.push(answers);
  
  if (state.answers.length < 5) {
    navigateToMeal();
  } else {
    // Send final state to Google Form before navigating to thanks page
    await sendToGoogleForm();
    navigateToThanks();
  }
}

/**
 * Sends the final state data to a Google Form
 */
async function sendToGoogleForm() {
  try {
    // Replace with your actual Google Form URL
    // The URL should be the form's "formResponse" endpoint
    const googleFormUrl = "https://docs.google.com/forms/d/e/1FAIpQLSfLnBKpCIQQxVQ44__VRmmsQveIIDsWFEgBioKYcoFBqh2KOA/formResponse";
    
    // Prepare the data to send
    // You need to use the actual field names from your Google Form
    // These are typically named "entry.XXXXXXX" where X is a number
    const formData = new FormData();
    
    // Add the compressed state as a single field
    // Replace "entry.XXXXXXX" with your actual form field ID
    formData.append("entry.437529290", compressState(state.answers));
    
    // Alternatively, you can add each answer separately if you have multiple form fields
    // Example:
    // state.answers.forEach((mealAnswer, index) => {
    //   formData.append(`entry.XXXXX.${index}`, JSON.stringify(mealAnswer));
    // });
    
    // Send the data using fetch API with POST method
    const response = await fetch(googleFormUrl, {
      method: "POST",
      mode: "no-cors", // Google Forms requires this
      body: formData
    });
    
    return true;
  } catch (error) {
    console.error("Error sending data to Google Form:", error);
    return false;
  }
}

/***********************************************************
 * INIT
 ***********************************************************/
// Initialize state from URL on page load
const { answers } = getStateFromUrl();
state.answers = answers;

// Set up event listeners based on current page
document.addEventListener('DOMContentLoaded', () => {
  // We don't need to set up event listeners for meal.html here anymore as it's handled in meal.html
  if (window.location.pathname.endsWith('index.html') || window.location.pathname.endsWith('/')) {
    // Handle the start button on the index page
    document.getElementById('start-btn').addEventListener('click', () => {
      navigateToMeal();
    });
  }
});

// Add browser back/forward button support
window.addEventListener('popstate', () => {
  const { answers } = getStateFromUrl();
  state.answers = answers;
  if (window.location.pathname.endsWith('meal.html')) {
    showMeal();
  }
}); 