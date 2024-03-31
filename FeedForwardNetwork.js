class FeedForwardNetwork {
	
	 constructor(){
		this.inputLayerSize = 784;
		this.hiddenLayerSize = 89;
		this.numHiddenLayers = 1;
		this.outputLayerSize = 10;
		this.maxLayerSize = Math.max(Math.max(this.inputLayerSize, this.hiddenLayerSize), this.outputLayerSize);

        this.desiredOutput = [];
        this.weights = WEIGHTS; // [layer][from][to]
	}

	sigmoidActivationFunction(input){
		return 1.0 / (1 + Math.exp(-1.0 * input));
	}
	
	feedForward(activation, fromLayerSize, toLayerSize, l){
		let inJ;
		for (let j = 0; j < toLayerSize; j++){
			inJ = 0;
			for (let i = 0; i < fromLayerSize; i++){
				inJ += this.weights[l][i][j] * activation[l][i];
			}
			// 0 is the first hidden layer
			activation[l+1][j] = this.sigmoidActivationFunction(inJ);
		}
	}
	
	
	testNetwork(inputs){
        let activation = [];
        for(let i = 0; i < this.numHiddenLayers+2; i++)
            for (let j = 0; j < this.maxLayerSize; j++)
                activation.push([]);

        for (let i = 0; i < this.inputLayerSize; i++){
            activation[0][i] = inputs[i];
        }

        // There may be different sizes for the input, hidden and output layers, hence there are three different calls for feedForeard
        // input to first hidden layer
        this.feedForward(activation, this.inputLayerSize, this.hiddenLayerSize, 0);
        // hidden to hidden layers
        for (let l = 1; l < this.numHiddenLayers; l++){
            this.feedForward(activation, this.hiddenLayerSize, this.hiddenLayerSize, l);					
        }
        // last hidden layer to output layer
        this.feedForward(activation, this.hiddenLayerSize, this.outputLayerSize, this.numHiddenLayers);
        return activation[this.numHiddenLayers+1];
	}
}
	

