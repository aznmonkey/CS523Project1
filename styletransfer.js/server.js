var express = require('express');
var app = express();
var path = require('path');
var formidable = require('formidable');
var fs = require('fs');
// Use python shell
var PythonShell = require('python-shell');

var uploadPath = '/uploads';
var processedPath = '/processed';

var pythonExecutable = '/home/evl/anishi2/bin/python3.5';
var pythonScriptPath = "/data/evl/anishi2/cs523/neural-style/";
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
	
  // every time a file has been uploaded successfully,
  // rename it to it's orignal name
  form.on('file', function(field, file) {
    fs.rename(file.path, path.join(form.uploadDir, file.name));
	
	const spawn = require('child_process').spawn;
	
	// Lyra SSH version
	pythonArgs = 
	[
		"lyra-02",
		"nohup",
		pythonExecutable,
		pythonScriptPath+pythonScript,
	 	"--content", form.uploadDir+"/"+file.name,
	 	"--styles", pythonScriptPath+"style1.jpg",
	 	"--output", processedImageDir+"/"+"testOutput.jpg",
	 	"--iterations", 100,
		"--network", pythonNetworkPath
	]
	const scriptExecution = spawn("ssh", pythonArgs);
	
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

	// pythonOptions.args = [
	// 	"--content " + file.name,
	// 	"--style " + "style1.jpg",
	// 	"--output " + "testOuput.jpg",
	// 	"--iterations " + "100"
	// ];
	// PythonShell.run(pythonScript, pythonOptions, function (err, results) {
	// 	if (err) throw err;
	// 	// results is an array consisting of messages collected during execution
	// 	console.log('results: %j', results);
	// });

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
