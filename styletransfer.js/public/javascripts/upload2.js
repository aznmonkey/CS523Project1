(function() {
  let contentFile = null,
      styleFile = null;

  function previewImage(img, selector) {
    var fr  = new FileReader();

    fr.addEventListener("load", function () {
      selector.style.backgroundImage = 'url(' + fr.result + ')';
    }, false);

    if (img.type.startsWith("image/")) {
      fr.readAsDataURL(img);      
    }
  }

  document.getElementById('content-input').onchange = function() {
    if (this.files[0]) {
      contentFile = this.files[0];
      previewImage(contentFile, document.getElementById('content-input-preview'));
    }
  }

  document.getElementById('style-input').onchange = function() {
    if (this.files[0]) {
      styleFile = this.files[0];
      previewImage(styleFile, document.getElementById('style-input-preview'));
    }
  }

  document.getElementById('render').onclick = function() {
    if (!contentFile || !styleFile) {
      document.querySelector('#renderdiv div').style.visibility = 'visible';
    }
    else {
      // upload

      // create a FormData object which will be sent as the data payload in the
      // AJAX request
      var formData = new FormData();

      formData.append('content',contentFile, contentFile.name);//{name: contentFile.name, type: 'content'});
      formData.append('style',styleFile, styleFile.name);//{name: styleFile.name, type: 'style'});

      $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data){
            console.log('upload successful!\n' + data);
        },
        xhr: function() {
          // create an XMLHttpRequest
          var xhr = new XMLHttpRequest();

          var progressBar = document.getElementById('progress');

          // listen to the 'progress' event
          xhr.upload.addEventListener('progress', function(evt) {

            if (evt.lengthComputable) {
              // calculate the percentage of upload completed
              var percentComplete = evt.loaded / evt.total;
              percentComplete = parseInt(percentComplete * 100);

              // update the Bootstrap progress bar with the new percentage
              progressBar.style.width = percentComplete + '%';

              // once the upload reaches 100%
              if (percentComplete === 100) {
                console.log('done uploading')
              }

            }

          }, false);

          return xhr;
        }
      });


    }
  }


  var socket = io.connect();
  socket.on('fileUploaded', function(data) {
    console.log(data);
  })

})();
