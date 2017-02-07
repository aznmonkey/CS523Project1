var express = require('express');
var app = express();
var path = require('path');
var formidable = require('formidable');
var fs = require('fs');
var socketio = require('socket.io');
// Use python shell
var PythonShell = require('python-shell');

var uploadPath = '/uploads';
var processedPath = '/processed';

//   /* arthur's paths */
var pythonExecutable = '/home/evl/anishi2/bin/python3.5';
var pythonScriptPath = "/data/evl/anishi2/cs523/CS523Project1/";

/* kristine's paths */
//var pythonExecutable = "/usr/bin/python"
//var pythonScriptPath = "/Users/kristinelee/Desktop/class/523/p1/neural-style-master/";
var pythonScript = "neural_style.py"
var pythonNetworkPath = pythonScriptPath+"imagenet-vgg-verydeep-19.mat"

// Set environment variables
// LD_LIBRARY_PATH is not read by default? (not needed if using ssh)
//process.env.LD_LIBRARY_PATH = "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64";

// Function to convert an Uint8Array to a string
var uint8arrayToString = function(data){
    return String.fromCharCode.apply(null, data);
};

var pythonOptions = {
	mode: 'text',
    pythonPath: pythonExecutable,
    pythonOptions: ['-u'],
    scriptPath: pythonScriptPath,
    args: []
};

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', function(req, res){
    res.sendFile(path.join(__dirname, 'views/index.html'));
});

app.post('/upload', function(req, res){

  // create an incoming form object
  var form = new formidable.IncomingForm();

  // specify that we want to allow the user to upload multiple files in a single request
  form.multiples = true;

  // store all uploads in the /uploads directory
  form.uploadDir = path.join(__dirname, uploadPath);
	
  var processedImageDir = path.join(__dirname, processedPath);

  var contentPath = null;
  var stylePath = null;

  // every time a file has been uploaded successfully,
  // rename it to it's orignal name
  form.on('file', function(field, file) {
    var newPath = form.uploadDir + '/' + field + '/' + file.name;
    fs.rename(file.path, newPath);
    if (field == 'content') {
        contentPath = newPath;
    }
    else {
        stylePath = newPath;
    }
	
	const spawn = require('child_process').spawn;

    if (!contentPath || !stylePath) { 
        return; 
    }
	
	// Lyra SSH version
	pythonArgs = 
	[
		"lyra-02",
		"nohup",
		pythonExecutable,
		pythonScriptPath+pythonScript,
	 	"--content", contentPath,
	 	"--styles", stylePath,
	 	"--output", processedImageDir+"/"+file.name,
	 	"--iterations", 100,
		"--network", pythonNetworkPath
	]
	
	console.log('running...');
	console.log(pythonArgs);
//   /* running on lyra */
   const scriptExecution = spawn("ssh", pythonArgs);

    /* running on local */
    //const scriptExecution = spawn("python", pythonArgs.slice(3));
	
	// Handle normal output
	scriptExecution.stdout.on('data', (data) => {
		console.log(uint8arrayToString(data));
	});

	// Handle error output
	scriptExecution.stderr.on('data', (data) => {
		// As said before, convert the Uint8Array to a readable string.
		console.log(uint8arrayToString(data));
	});

	scriptExecution.on('exit', (code) => {
		console.log("Process quit with code : " + code);
	});

  });

  // log any errors that occur
  form.on('error', function(err) {
    console.log('An error has occured: \n' + err);
  });

  // once all the files have been uploaded, send a response to the client
  form.on('end', function() {
    res.end('success');
  });

  // parse the incoming request containing the form data
  form.parse(req);

});

var server = app.listen(10523, function(){
  console.log('Server listening on port 10523');
});

const io = socketio(server);
io.on('connection', function(socket) {
    fs.watch(path.join(__dirname, processedPath), (evt, filename) => {

      if (filename) {
        fs.readFile(path.join(__dirname, processedPath) + '/' + filename, function(err, data) {
            if (err) {
                console.log(err);
            }
            else {
                socket.emit('fileUploaded', { file: processedPath + '/' + filename, image: data });
            }
        })
      } else {
        console.log('filename not provided');
      }
    });    
})

