async function submitForm() {
    var formData = new FormData(document.getElementById('myForm'));

    const inputName = document.getElementById('inputName').value;
    const inputNumber = document.getElementById('inputNumber').value;
    const inputAadhar = document.getElementById('inputAadhar').value;

    console.log(inputAadhar);
    console.log(inputNumber);
    console.log(inputName);
 
    var btn = document.getElementsByClassName("btn")[0];
    btn.disabled = true;
    const spinner = document.getElementById("spinner");
    spinner.classList.remove("d-none");
    const loadImageBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = (error) => reject(error);
        });
    }
    const fileInput = document.getElementById('fileInput');
    // Check if a file is selected
    if (fileInput.files.length > 0) {
        // Pass the selected file to loadImageBase64 function
        const image = await loadImageBase64(fileInput.files[0]);
        const data={
            "inputAadhar":inputAadhar,
            "inputName":inputName,
            "inputNumber":inputNumber,
            "image":image
        }
        axios({
            method: "POST",
            url: "/submit",
            data: data ,
            headers: {
                "Content-Type": "application/json",
            },
        })
        .then(function(response) {
            console.log(response.data)
            var resultImage = document.getElementById('resultImage');
            var list_class = document.getElementById('list_class');
            var bar = document.getElementById('bar');
            var bar_value = document.getElementById('bar_value');
            spinner.classList.add("d-none");
            list_class.classList.remove("d-none");
            bar.classList.remove("d-none");
            resultImage.src = 'data:image/jpeg;base64,'+response.data.roboflow_result;
            let percentage=0;
            for (let key in response.data.details_set) {
                if (response.data.details_set.hasOwnProperty(key)) {
                    const i = response.data.details_set[key];
                    console.log(i + percentage);
                    if (key !== "aadharlogo") {
                        percentage++;
                        if(i==="True") document.getElementById(key).src = "/static/correct.gif";
                        document.getElementById(key + "-status").innerHTML = `<img id="" src="/static/${key}.jpg" alt="${key} Image">`;
                    }
                }
            }
            
            bar_value.style.width=(percentage/6)*100+'%'
            bar_value.innerHTML=(percentage/6)*100+'%'
            btn.disabled = false;
        })
        .catch(function(error) {
            console.error('Error:', error);
        });
    } else {
        console.error('No file selected');
    }
}
