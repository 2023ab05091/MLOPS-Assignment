const form = document.getElementById("values");

form.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  const formData = new FormData(form);
  const result = await submitValues(formData);
  setResultOutput(result);
});

const submitValues = async (formData) => {
  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    return result;
  } catch (error) {
    console.error(error);
  }
};

const setResultOutput = (result) => {
  const output = document.getElementById("output");
  output.textContent = result.text;
  output.style.display = "block";
  if (result.status === "safe") {
    output.classList.remove("red-text");
    output.classList.add("green-text");
  } else {
    output.classList.remove("green-text");
    output.classList.add("red-text");
  }
};
