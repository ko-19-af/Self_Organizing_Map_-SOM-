import BasicIO.ASCIIDataFile;
import java.util.Random;
import java.util.Stack;
/** This class uses the SOM AI to cluster data
 *
 * IDE used: IntelliJ
 *
 * @Author Kevin Olenic
 * ST#: 6814974
 * @Version 2.3
 * @Since 2023-04-09
 */
public class SOM {
    private final static double alpha = 1;// learning rate at t(0)
    private static final int sigma = 5;// neighbourhood range at t(0)
    private final static int e = 2000;// number of epochs
    private final static Random r = new Random();// for generating random numbers
    private static int[] Topology;// array to hold the topology of the SOM
    private static double[] ignitionData;// holds data on engine ignition status
    private static double[][] reelData;// holds data on engine reel data
    private static double[][] vectors;// array holding the vectors of the individual nodes


    public SOM(){
        readFile(new ASCIIDataFile());// load first files data
        Topology = new int[] {49, reelData[0].length};// topology of SOM (# of clusters, # of inputs) (25,36,49)
        initialize();// initialize weights of clusters

        for(int x = 0; x <= e; x++){// for each epoch
            int random;
            double nr = Math.floor(sigma * Math.exp(-x/2000.0));// adjust the neighbourhood size each epoch
            double lr = alpha * Math.exp(-x/2000.0);// adjust learning rate for next epoch
            Stack<Integer> stack = new Stack<>();// stack for holding already processed
            for (double[] ignored : reelData) {// for each training data (randomly select one)
                do {
                    random = (int) (Math.random() * reelData.length);//randomly choose training data set
                } while (stack.contains(random));
                stack.add(random);// add used training data to stack
                update(reelData[random], winner(reelData[random]), lr, nr);// update the weight based on the winning cluster
            }
        }
        gHeatmap();// calculate the heat at each node in SOM map
        System.exit(1);
    }

    public static void main(String[] args) {new SOM();}// main

    /**This function initializes the vector values of the SOM by applying one of the inputs weights to the values
     *
     */
    private void initialize(){
        vectors = new double[Topology[0]][Topology[1]];// initialize SOM nodes vectors table
        Stack<Integer> used = new Stack<>();// stack for holding already chosen input data
        for(int y = 0; y < Topology[0]; y++){// for every cluster
            int rando;
            if(used.empty()) rando = r.nextInt(reelData.length);// randomly choose input data
            else {
                do {
                    rando = r.nextInt(reelData.length);// randomly choose input data
                } while (used.contains(rando));
            }used.add(rando);// add randomly chosen data to stack
            double[] copy = reelData[rando];// copy data from input data
            System.arraycopy(copy, 0, vectors[y], 0, copy.length);// set vectors to value of input data
        }
    }//initialize

    /** This method determines which neuron in the SOM the input belongs to based on the euclidean distance
     * between the two
     *
     * @param sample is the input being classified
     * @return the winning neuron number
     */
    private int winner(double[] sample){
        int BMU = 0; // best matching unit
        double lowestDistance = Double.MAX_VALUE;// initialize lowest distance variable
        for(int y = 0; y < Topology[0]; y++){// go through each cluster
            double distance = 0;// get distance of cluster from an input
            for(int x = 0; x < Topology[1]; x++){// go through the weights
                distance += Math.pow((sample[x] - vectors[y][x]), 2);// get euclidean distance
            }
            distance = Math.sqrt(distance);
            if(distance < lowestDistance){// if distance of current node is better than best matching unit
                lowestDistance = distance;// update to best distance
                BMU = y;// assign BMU as best matching unit
            }
        }
        return BMU;// return winning neuron
    }//winner

    /**This method updates the weights of the neurons in the SOM
     *
     * @param sample is the input being classified
     * @param winner is the node the input is being classified to
     * @param alpha is the learning rate at time t
     * @param lattice current neighbourhood size at time t
     */
    private void update( double[] sample, int winner, double alpha, double lattice){
        for(int y = 0; y < vectors.length; y++) {// For each node in the topology
            double nInfluence = influence(winner, y, lattice);// calculate neighbourhood influence
            for (int x = 0; x < vectors[0].length; x++){// for every feature vector in training sample
                vectors[y][x] += alpha * nInfluence * (sample[x] - vectors[y][x]);// update weights
            }
        }
    }//update

    /**This class calculates the influence on a node in the neighbourhood using the gaussian function
     *
     * @param winner is the neuron the input is being classified
     * @param neuron is the unit in the network being adjusted
     * @param lattice is the bound of the neighbourhood at time t
     * @return the influence on the node in the neighbourhood
     */

    private double influence(int winner, int neuron, double lattice){
        double distance = (-1) * Math.pow(pyDist(winner, neuron),2);// distance between nodes in topology
        double range = 2 * Math.pow(lattice,2);// calculate range of neighbourhood
        return Math.exp(distance/range);// return distance (Gaussian function)
    }// influence

    /** This method calculates the distance from the winning neuron to another neuron in the topology
     * using pythagorean distance
     *
     * @param w is the wining neuron number
     * @param n is the number of the other neuron
     * @return the distance of the neuron from the winning neuron
     */
    private double pyDist(int w, int n){
        int wx = w % (int)Math.sqrt(Topology[0]);// get x co-ordinate of winning neuron in topology
        int wy = (w-wx) / (int)Math.sqrt(Topology[0]);// initialize y co-ordinate of winning neuron in topology
        int nx = n % (int) Math.sqrt(Topology[0]);// get x co-ordinate of other neuron in topology
        int ny = (n-nx) / (int)Math.sqrt(Topology[0]);// get y-co-ordinate of neuron in topology

        // create wrapping
        if(wx == nx) {// if they have same x value
            if (wy == 0 && ny == Math.sqrt(Topology[0]) - 1) return 1;// wrap top with bottom
            else if(ny == 0 && wy == Math.sqrt(Topology[0]) - 1) return 1;// wrap bottom with top
        }else if(wy == ny){// if they have the same y value
            if (wx == 0 && nx == Math.sqrt(Topology[0]) - 1) return 1;// wrap left with right
            else if(nx == 0 && wx == Math.sqrt(Topology[0]) - 1) return 1;// wrap right with left
        }
        // if nodes don't wrap calculate pythagorean distance
        int y = Math.abs(wy-ny);// get y distance from winning node
        int x = Math.abs(wx-nx);// get x distance from winning node
        return Math.sqrt(Math.pow(x,2) + Math.pow(y,2));// calculate pythagorean distance between nodes in topology
    }//pyDist

    /**This method generates the heatmap for the SOM
     *
     */
    private void gHeatmap(){
        for(int a = 0; a < Topology[0]; a++){// for each cluster
            double totalHeat = 0;// initialize heat of cluster being examined
            for (int y = 0; y < reelData.length; y++){// for each input vector
                double distance = 0;
                for (int x = 0; x < reelData[0].length; x++) {// for each feature in vector
                    distance += Math.pow(reelData[y][x] - vectors[a][x], 2);// calculate euclidean distance
                }
                distance = (-1) * Math.pow(Math.sqrt(distance),2);
                double neighbourhood = 2 * Math.pow(15, 2);
                distance = Math.exp(distance/neighbourhood);// calculate activation
                if(ignitionData[y] == 0) totalHeat += distance;// if reading good engine
                else totalHeat -= distance;// if reading bad engine
            }
            if(a % Math.sqrt(Topology[0] ) == (int)Math.sqrt(Topology[0])-1) System.out.print(totalHeat+"\n");// start new line
            else System.out.print(totalHeat + "\t");// write heat for node in topology
        }
    }//gHeatmap

    /** This method reads a data file and collects the information on it and normalizes the data
     * @param file containing the data being read
     */

    private void readFile(ASCIIDataFile file){
        String[] data = file.readString().split(" ");
        ignitionData = new double[Integer.parseInt(data[0])];
        reelData = new double[Integer.parseInt(data[0])][Integer.parseInt(data[1])];
        for(int y = 0; y < Integer.parseInt(data[0]); y++){// read all data entries
            String[] eData = file.readString().split(" ");// read engine data
            ignitionData[y] = Integer.parseInt(eData[0]);// load ignition data
            int position = 0;// position in reelData
            for(int x = 1; x < eData.length; x++){//load reel data
                if(!eData[x].equals("")) {// if valid data to enter into array
                    double value = Double.parseDouble(eData[x]);
                    reelData[y][position] = value;// from data read from file
                    position++;// move to next position when data is added
                }
            }
        }
        file.close();// close the file
    }//readFile
}// SOM