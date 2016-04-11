-----------------------------------
--
--     Tutorial Deep Learning at Telecom Paristech
--     Instructor: Gaetan Marceau Caron - Equipe TAO (LRI, INRIA-Saclay, CNRS) (gaetan.marceau-caron@inria.fr)
--     04/08/2016
--
-----------------------------------
require 'optim'
require 'nn'
require 'cutorch'
require 'cunn'

-- If you want to try the Riemannian algorithms:
-- uncommment the following "require ...",
-- replace nn.Linear with nn.RDLayer or nn.RQDLayer
-- RQDLayer should be better

--require './rdLayer.lua'
--require './rqdLayer.lua'

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
   cmd:option('-learningRate', 1, 'learning rate at t=0')
   cmd:option('-batchSize', 500, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-maxEpoch', 100, 'maximum nb of epoch')
   cmd:option('-seed', 0, 'random seed')
   cmd:option('-saveModel', false, 'flag for saving the model on disk at each epoch, if improvement')
   cmd:option('-plot',false,'flag for plotting valid loss and train loss as a function of epochs')
   cmd:option('-gpu',false,'flag for using GPU')
   cmd:option('-alpha', 0.01, 'alpha at t=0')
   cmd:option('-epsilon', 0.01, 'epsilon at t=0')
   cmd:option('-dropout', false, 'apply dropout')
  
   cmd:text()
   opt = cmd:parse(arg or {})
end
torch.manualSeed(opt.seed)

if opt.saveModel then
	os.execute('rm -rf model/')
end

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
confusion = optim.ConfusionMatrix(classes)
trainLogger = optim.Logger('train.log')
testLogger = optim.Logger('test.log')
timeLogger = optim.Logger('time.log')

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
if opt.dropout then
	model:add(nn.Dropout(0.2))
end
model:add(nn.Linear(nInput,400))
if opt.dropout then
	model:add(nn.Dropout(0.5))
end
model:add(nn.Tanh())
model:add(nn.Linear(400,400))
if opt.dropout then
	model:add(nn.Dropout(0.5))
end
model:add(nn.Tanh())
model:add(nn.Linear(400,10))
model:add(nn.LogSoftMax())
-- ... Complete here ...
---------------------------

---------------------------
-- Definition of the criterion
criterion = nn.ClassNLLCriterion()
--------------------------

-- Retrieve the pointers to the parameters and gradParameters from the model for latter use
parameters,gradParameters = model:getParameters()
--
--[[
if opt.gpu then
	targets = targets:cuda()
	inputs = inputs:cuda()
	trainSet = trainSet:cuda()
	validSet = validSet:cuda()
	criterion:cuda()
	model:cuda()
end
]]--

-- Learning function
function train()

   local tick1 = sys.clock()
   print()

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
		 --print(parameters:size())
		 --print(x:size()) 
		 if parameters~=x then
			parameters:copy(x)
		 end
		 
		 gradParameters:zero()

		 --------------------------------------------------------
		 -- Complete the main steps for training a neural network
		 -- ... complete here ...
		 local outputs = model:forward(inputs)
		 local f = criterion:forward(outputs,targets)
		 local df_fw = criterion:backward(outputs,targets)
		 model:backward(inputs,df_fw)
		 --------------------------------------------------------

		 return parameters,gradParameters
		 
	  end
	  optim.sgd(feval,parameters,opt)
	  --optim.rmsprop(feval,parameters,opt)
   end
   time_epoch = sys.clock()-tick1
   print("tick " .. time_epoch .. 's')
   print('\n')
end

tbegin = sys.clock()
print '==> training'
prevLoss = 10e12
for i = 1,opt.maxEpoch do
   
   -- Evaluating the model
   print("EPOCH: " .. i)
   ------------------------------
   -- complete here, if necessary
   ------------------------------

   if opt.dropout then
   	model:evaluate()
   end
   confusion:zero()
   local trainPred = model:forward(trainSet)
   local trainLoss = criterion:forward(trainPred, trainSetLabel) 
   local totalLost = 0
   for i = 1,trainPred:size(1) do
	  confusion:add(trainPred[i],trainSetLabel[i])
   end
   print(confusion)
   trainLogger:add{['% mean class error (train set)'] = 100*(1-confusion.totalValid)}
   print(" + Train loss " .. trainLoss)-- .. totalLost/trainPred:size(1))

   confusion:zero()
   local validPred = model:forward(validSet)
   local validLoss = criterion:forward(validPred, validSetLabel) 
   for i = 1,validPred:size(1) do
	  confusion:add(validPred[i],validSetLabel[i])
   end
   print(confusion)
   testLogger:add{['% mean class error (test set)'] = 100*(1-confusion.totalValid)} 
   print(" + Valid loss " .. validLoss)

   ------------------------------
   -- complete here, if necessary
   ------------------------------
   if validLoss<prevLoss then
	if opt.saveModel and i==opt.maxEpoch then
		local filename = paths.concat('model', 'mnist_epoch_' .. i .. '.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
        	-- if paths.filep(filename) then
			-- os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		torch.save(filename,model)
		print('<trainer> saving network to ' .. filename)
	end
   end
   if opt.plot then
	trainLogger:style{['% mean class error (train set)'] = '-'}
	testLogger:style{['% mean class error (test set)'] = '-'}
	trainLogger:plot()
	testLogger:plot()
   end
   prevLoss = validLoss
   if opt.dropout then
   	model:training()
   end
   train()
end
tend = sys.clock()
print('total time : ' .. tend-tbegin .. 's')
timeLogger:add{['total time (s) : '] = tend-tbegin} 
