const videoSubject = document.querySelector("#videoSubject");
const videoInputType = document.querySelector("#videoInputType");
const videoType = document.querySelector("#videoType");
const aiModel = document.querySelector("#aiModel");
const youtubeToggle = document.querySelector("#youtubeUploadToggle");
const useMusicToggle = document.querySelector("#useMusicToggle");
const instagramToggle = document.querySelector("#instagramUploadToggle");
const customPrompt = document.querySelector("#customPrompt");
const generateButton = document.querySelector("#generateButton");
const cancelButton = document.querySelector("#cancelButton");
const scriptContainer = document.querySelector("#scriptContainer");
const scriptContent = document.querySelector("#scriptContent");
const generateScriptButton = document.querySelector("#generateScriptButton");

const advancedOptionsToggle = document.querySelector("#advancedOptionsToggle");

const scriptLoader = document.querySelector("#scriptLoader");

const linkInput = document.querySelector("#linkInput");
const linkInputField = document.querySelector("#linkInputField");

videoInputType.addEventListener("change", function () {
  if (this.value === "youtubeURL" || this.value === "blogLink") {
    linkInput.style.display = "block";
  } else {
    linkInput.style.display = "none";
    linkInputField.value = "";
  }
});

advancedOptionsToggle.addEventListener("click", () => {
  // Change Emoji, from ▼ to ▲ and vice versa
  const emoji = advancedOptionsToggle.textContent;
  advancedOptionsToggle.textContent = emoji.includes("▼")
    ? "Show less Options ▲"
    : "Show Advanced Options ▼";
  const advancedOptions = document.querySelector("#advancedOptions");
  advancedOptions.classList.toggle("hidden");
});

const cancelGeneration = () => {
  console.log("Canceling generation...");
  // Send request to /cancel
  fetch("http://localhost:8080/api/cancel", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      alert(data.message);
      console.log(data);
    })
    .catch((error) => {
      alert("An error occurred. Please try again later.");
      console.log(error);
    });

  // Hide cancel button
  cancelButton.classList.add("hidden");

  // Enable generate button
  generateButton.disabled = false;
  generateButton.classList.remove("hidden");
};

const generateVideo = () => {
  console.log("Generating video...");
  // Disable button and change text
  generateButton.disabled = true;
  generateButton.classList.add("hidden");

  // Show cancel button
  cancelButton.classList.remove("hidden");

  // Get values from input fields
  const videoSubjectValue = videoSubject.value;
  const linkInputFieldValue = linkInputField.value;
  const linkInputValue = linkInput.value;
  const videoInputTypeValue = videoInputType.value;
  const aiModelValue = aiModel.value;
  const videoTypeValue = videoType.value;
  const youtubeUpload = youtubeToggle.checked;
  const useMusicToggleState = useMusicToggle.checked;
  const instagramUpload = instagramToggle.checked;
  const threads = document.querySelector("#threads").value;
  const subtitlesPosition = document.querySelector("#subtitlesPosition").value;
  const colorHexCode = document.querySelector("#subtitlesColor").value;
  const templates = document.querySelector("#templates").value;

  const script = scriptContent.textContent.trim();

  const url = "http://localhost:8080/api/generate";

  // Construct data to be sent to the server
  const data = {
    videoInputType: videoInputTypeValue,
    linkInput: linkInputValue,
    videoSubject: videoSubjectValue,
    linkInputField: linkInputFieldValue,
    videoType: videoTypeValue,
    aiModel: aiModelValue,
    automateYoutubeUpload: youtubeUpload,
    useMusic: useMusicToggleState,
    automateInstagramUpload: instagramUpload,
    threads: threads,
    subtitlesPosition: subtitlesPosition,
    color: colorHexCode,
    templates: templates,
    script: script,
  };

  // Send the actual request to the server
  fetch(url, {
    method: "POST",
    body: JSON.stringify(data),
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      alert(data.message);
      // Hide cancel button after generation is complete
      generateButton.disabled = false;
      generateButton.classList.remove("hidden");
      cancelButton.classList.add("hidden");
    })
    .catch((error) => {
      alert("An error occurred. Please try again later.");
      console.log(error);
    });
};

generateButton.addEventListener("click", generateVideo);
cancelButton.addEventListener("click", cancelGeneration);

videoSubject.addEventListener("keyup", (event) => {
  if (event.key === "Enter") {
    generateVideo();
  }
});

// Load the data from localStorage on page load
document.addEventListener("DOMContentLoaded", (event) => {
  const voiceSelect = document.getElementById("voice");
  const storedVoiceValue = localStorage.getItem("voiceValue");

  if (storedVoiceValue) {
    voiceSelect.value = storedVoiceValue;
  }
});

// Save the data to localStorage when the user changes the value
toggles = ["youtubeUploadToggle", "instagramUploadToggle", "useMusicToggle"];
fields = [
  "videoInputType",
  "linkInput",
  "videoType",
  "aiModel",
  "videoSubject",
  "linkInputField",
  "customPrompt",
  "threads",
  "subtitlesPosition",
  "subtitlesColor",
  "templates",
];

document.addEventListener("DOMContentLoaded", () => {
  toggles.forEach((id) => {
    const toggle = document.getElementById(id);
    const storedValue = localStorage.getItem(`${id}Value`);
    const storedReuseValue = localStorage.getItem("reuseChoicesToggleValue");

    if (toggle && storedValue !== null && storedReuseValue === "true") {
      toggle.checked = storedValue === "true";
    }
    // Attach change listener to update localStorage
    toggle.addEventListener("change", (event) => {
      localStorage.setItem(`${id}Value`, event.target.checked);
    });
  });

  fields.forEach((id) => {
    const select = document.getElementById(id);
    const storedValue = localStorage.getItem(`${id}Value`);
    const storedReuseValue = localStorage.getItem("reuseChoicesToggleValue");

    if (storedValue && storedReuseValue === "true") {
      select.value = storedValue;
    }
    // Attach change listener to update localStorage
    select.addEventListener("change", (event) => {
      localStorage.setItem(`${id}Value`, event.target.value);
    });
  });
});

const generateScript = () => {
  generateScriptButton.disabled = true;
  generateScriptButton.classList.add("hidden");

  // Show the loader
  scriptLoader.classList.remove("hidden");

  const videoSubjectValue = videoSubject.value;
  const linkInputFieldValue = linkInputField.value;
  const linkInputValue = linkInput.value;
  const videoInputTypeValue = videoInputType.value;
  const videoTypeValue = videoType.value;

  fetch("http://localhost:8080/api/generate-script", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      videoInputType: videoInputTypeValue,
      linkInput: linkInputValue,
      videoSubject: videoSubjectValue,
      linkInputField: linkInputFieldValue,
      videoType: videoTypeValue,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide the loader
      scriptLoader.classList.add("hidden");
      if (data.success) {
        scriptContent.innerHTML = data.script;
        scriptContainer.classList.remove("hidden");
        generateScriptButton.disabled = false;
        generateScriptButton.classList.remove("hidden");
        generateButton.classList.remove("hidden"); // Show the "Generate Video" button
      } else {
        alert("Failed to generate script. Please try again.");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred. Please try again later.");
      // Hide the loader
      scriptLoader.classList.add("hidden");
    });
};

generateScriptButton.addEventListener("click", generateScript);
