-----------------------------------
--
--     Tutorial Deep Learning at Telecom
--     Instructor: Gaetan Marceau Caron (gaetan.marceau-caron@inria.fr)
--     08/04/2016
--
-----------------------------------
require 'optim'
require 'nn'

-- If you want to try the Riemannian algorithms:
-- uncommment the following "require ...",
-- replace nn.Linear with nn.RDLayer or nn.RQDLayer
-- RQDLayer should be better

require './rdLayer.lua'
require './rqdLayer.lua'

--------------------------------
-- Parameter Handling
--------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Deep Learning - Telecom tutorial')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-learningRate', 0.000001, 'learning rate at t=0')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 100, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-gradient', "rqd", 'Gradient type: |normal|rd|rqd|')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.manualSeed(opt.seed)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
trainLogger:setNames({'epoch','trainLoss','trainAccuracy','validLoss','validAccuracy'})

--------------------------------------
-- Loading and normalizing the dataset
local nTrain = 50000
local nValid = 10000
local mnist = require 'mnist'
local mnistDataset = mnist.traindataset()
local nInput = mnistDataset.data:size(2) * mnistDataset.data:size(3)

-- classes
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- This matrix records the current confusion across classes
trainConfusion = optim.ConfusionMatrix(classes)
validConfusion = optim.ConfusionMatrix(classes)

trainSet = torch.Tensor(nTrain,mnistDataset.data:size(2),mnistDataset.data:size(3))
trainSet:copy(mnistDataset.data:narrow(1,1,nTrain):float():div(255.))
trainSetLabel = torch.Tensor(nTrain)
trainSetLabel:copy(mnistDataset.label:narrow(1,1,nTrain))
trainSetLabel:add(1)

validSet = torch.Tensor(nValid,mnistDataset.data:size(2),mnistDataset.data:size(3))
validSet:copy(mnistDataset.data:narrow(1,nTrain+1,nValid):float():div(255.))
validSetLabel = torch.Tensor(nValid)
validSetLabel:copy(mnistDataset.label:narrow(1,nTrain+1,nValid))
validSetLabel:add(1)

inputs = torch.Tensor(opt.batchSize,trainSet:size(2),trainSet:size(3))
targets = torch.Tensor(opt.batchSize)   
-------------------------------------

---------------------------
-- Definition of the model
--
-- Complete the definition of your model here
model = nn.Sequential()
model:add(nn.Reshape(nInput))

if opt.gradient == "normal" then
   model:add(nn.Linear(nInput,800))
   model:add(nn.ReLU())
   model:add(nn.Linear(800,800))
   model:add(nn.ReLU())
   model:add(nn.Linear(800,10))
elseif opt.gradient == "rd" then
   model:add(nn.RDLayer(nInput,800))
   model:add(nn.ReLU())
   model:add(nn.RDLayer(800,800))
   model:add(nn.ReLU())
   model:add(nn.RDLayer(800,10))
else
   model:add(nn.RQDLayer(nInput,800))
   model:add(nn.ReLU())
   model:add(nn.RQDLayer(800,800))
   model:add(nn.ReLU())
   model:add(nn.RQDLayer(800,10))   
end
model:add(nn.LogSoftMax())

---------------------------
-- Definition of the criterion
criterion = nn.ClassNLLCriterion()
--------------------------

-- Retrieve the pointers to the parameters and gradParameters from the model for latter use
parameters,gradParameters = model:getParameters()
--

-- Learning function
function train()

   local tick1 = sys.clock()
   
   -- It may help to shuffle the examples
   shuffle = torch.randperm(trainSet:size(1))
   
   for t = 1,trainSet:size(1),opt.batchSize do
	  
	  xlua.progress(t,trainSet:size(1))
	  
	  -- Define the minibatch
	  for i = 1,opt.batchSize do
		 inputs[i]:copy(trainSet[shuffle[t+i-1]])
		 targets[i] = trainSetLabel[shuffle[t+i-1]]
	  end

	  -- Definition of the evaluation function (closure)
	  local feval = function(x)
		 
		 if parameters~=x then
			parameters:copy(x)
		 end
		 
		 gradParameters:zero()

		 --------------------------------------------------------
		 -- Complete the main steps for training a neural network
		 local outputs = model:forward(inputs)
		 local cost = criterion:forward(outputs, targets)
		 local dfdo = criterion:backward(outputs, targets)
		 model:backward(inputs,dfdo)
		 --------------------------------------------------------
		 
		 return parameters,gradParameters
		 
	  end
	  optim.sgd(feval,parameters,opt)
   end
   print("tick" .. sys.clock()-tick1)
end

prevLoss = 10e12
for i = 1,opt.maxEpoch do
   
   -- Evaluating the model
   model:evaluate()
   local trainPred = model:forward(trainSet)
   local trainLoss = criterion:forward(trainPred, trainSetLabel) 

   trainConfusion:batchAdd(trainPred, trainSetLabel)
   print("EPOCH: " .. i)
   print(trainConfusion)
   print(" + Train loss " .. trainLoss)
   
   local validPred = model:forward(validSet)
   local validLoss = criterion:forward(validPred, validSetLabel) 

   validConfusion:batchAdd(validPred, validSetLabel)
   print(validConfusion)
   print(" + Valid loss " .. validLoss)

   trainLogger:add{i, trainLoss, trainConfusion.totalValid * 100, validLoss, validConfusion.totalValid * 100}
   trainConfusion:zero()
   validConfusion:zero()

   if opt.saveModel then
	  if trainLoss < prevLoss then
		 prevLoss = trainLoss
		 torch.save("model.bin",model)
	  else
		 model = torch.load("model.bin")
	  end
   end
   
   model:training()
   train()
end
