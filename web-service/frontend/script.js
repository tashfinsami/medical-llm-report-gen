const imageInput = document.getElementById("imageInput");
const generateButton = document.getElementById("generateButton");
const downloadButton = document.getElementById("downloadButton");
const reportElement = document.getElementById("report");
const statusElement = document.getElementById("status");

let reportText = "";


generateButton.addEventListener("click", async () => {

    document.getElementById("error-container").hidden = true;

    const file = imageInput.files[0];

    if (!file) {
        statusElement.textContent = "Please select an image.";
        return;
    }

    const formData = new FormData();

    formData.append("file", file);

    generateButton.disabled = true;
    downloadButton.hidden = true;

    reportElement.textContent = "";
    statusElement.textContent = "Generating report...";

    try {

        const response = await fetch(
            "/generate-report",
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error(
                `Server returned ${response.status}`
            );
        }

        reportText = await response.text();

        //reportElement.textContent = reportText;
        reportElement.innerHTML = marked.parse(reportText); // for markdown data

        statusElement.textContent = "Report generated successfully.";

        downloadButton.hidden = false;

    } catch (error) {

        document.getElementById("error-container").hidden = false;

        console.error(error);

        statusElement.textContent =
            "Failed to generate the report.";

    } finally {

        generateButton.disabled = false;

    }
});


downloadButton.addEventListener("click", () => {

    const blob = new Blob(
        [reportText],
        { type: "text/markdown" }
    );

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;
    link.download = "mammogram_report.md";

    link.click();

    URL.revokeObjectURL(url);
});