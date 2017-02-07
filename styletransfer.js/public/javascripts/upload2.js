(function() {
  let newFileContainer = document.getElementById('newFile');

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
      newFileContainer.style.display = 'none';
      contentFile = this.files[0];
      previewImage(contentFile, document.getElementById('content-input-preview'));
    }
  }

  document.getElementById('style-input').onchange = function() {
    if (this.files[0]) {
      newFileContainer.style.display = 'none';
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


/* receiving image from server */
    function arrayBufferToDataUri(arrayBuffer) {
        var base64 = '',
            encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/',
            bytes = new Uint8Array(arrayBuffer), byteLength = bytes.byteLength,
            byteRemainder = byteLength % 3, mainLength = byteLength - byteRemainder,
            a, b, c, d, chunk;

        for (var i = 0; i < mainLength; i = i + 3) {
            chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
            a = (chunk & 16515072) >> 18; b = (chunk & 258048) >> 12;
            c = (chunk & 4032) >> 6; d = chunk & 63;
            base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d];
        }

        if (byteRemainder == 1) {
            chunk = bytes[mainLength];
            a = (chunk & 252) >> 2;
            b = (chunk & 3) << 4;
            base64 += encodings[a] + encodings[b] + '==';
        } else if (byteRemainder == 2) {
            chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1];
            a = (chunk & 16128) >> 8;
            b = (chunk & 1008) >> 4;
            c = (chunk & 15) << 2;
            base64 += encodings[a] + encodings[b] + encodings[c] + '=';
        }
        return "data:image/jpeg;base64," + base64;
    }

  var socket = io.connect();
  socket.on('fileUploaded', function(data) {
    if (data.file) {

      let newFileContainer = document.getElementById('newFile');
      newFileContainer.innerHTML = '<img src="' + arrayBufferToDataUri(data.image) + '">';
      newFileContainer.style.display = 'block';

    }
  })

})();
