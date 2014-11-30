package org.assign4;

import java.awt.Color;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.assign4.PrimGridWorldDomain.Point;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D.PolicyGlyphRenderStyle;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.QComputablePlanner;
import burlap.behavior.singleagent.planning.StateConditionTest;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.deterministic.TFGoalCondition;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.behavior.statehashing.StateHashFactory;
import burlap.behavior.statehashing.StateHashTuple;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldStateParser;
import burlap.oomdp.auxiliary.StateParser;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectInstance;
import burlap.oomdp.core.State;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.singleagent.ActionObserver;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.SinglePFTF;

public class GridWorldTester2 {

	PrimGridWorldDomain gridWorld;
	Domain domain;
	StateParser sp;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	DiscreteStateHashFactory hashingFactory;
	StateSpaceObserver stateSpaceObserver;

	public static void main(String[] args) {

		// -2.01 vs -0.01 - vi more sensitive to cost vs pi?
		// 4 loc, no neg, -.01 cost; poor maximization

		// double rewardTotal = 0;
		int iterations = 1000;
		AnalysisData viData = new AnalysisData(0, 0, 0, 0., 0);
		AnalysisData piData = new AnalysisData(0, 0, 0, 0., 0);
		AnalysisData qData = new AnalysisData(0, 0, 0, 0., 0);
		for (int i = 0; i < iterations; i++) {
			GridWorldTester2 test = new GridWorldTester2(-0.1);
			// System.out.println(String.format("# states: %d",
			// example.gridWorld.countStates()));
			String outputPath = "output/";

			// uncomment the example you want to see (and comment-out the rest)
			int numStates = test.gridWorld.countStates();
			AnalysisData ad = test.runValueIteration(outputPath);
			addData(viData, ad, numStates);
			ad = test.runPolicyIteration(outputPath);
			addData(piData, ad, numStates);
			ad = test.runQLearning(outputPath);
			addData(qData, ad, numStates);
		}
		printAverages("VI - ", viData, iterations);
		printAverages("PI - ", piData, iterations);
		printAverages("Q - ", qData, iterations);
	}

	private static void printAverages(String label, AnalysisData data, double num) {
		double avgTime = data.elapsed / num;
		double avgReward = data.reward / num;
		double avgStates = data.totalStates / num;
		double maxRewardRate = data.maxRewardCount / num;
		double avgIterations = data.iterationsUntilConversion / num;
		System.out.println(String.format(
				"%s Reward: %.3f, States: %.3f, Time: %.1f, State Exploration: %.3f, Max rate: %.3f, Iterations: %.1f", label, avgReward,
				avgStates, avgTime, ((double) data.uniqueStates / (double) data.totalStates), maxRewardRate, avgIterations));
	}

	private static void addData(AnalysisData data, AnalysisData ad, int numStates) {
		data.totalStates += numStates;
		data.totalStateVisits += ad.totalStateVisits;
		data.uniqueStates += ad.uniqueStates;
		data.reward += ad.reward;
		data.maxRewardCount += ad.maxRewardCount;
		data.elapsed += ad.elapsed;
		data.iterationsUntilConversion += ad.iterationsUntilConversion;
	}

	public GridWorldTester2(double defaultCost) {

		File outputDir = new File("./output");
		if (outputDir.exists()) {
			File[] files = outputDir.listFiles();
			if (files != null) {
				for (File f : files) {
					if (f.getName().endsWith(".episode")) {
						f.delete();
					}
				}
			}
		}

		// create the domain
		gridWorld = new PrimGridWorldDomain(5, 5);
		
		gridWorld.makeEmptyMap();
		gridWorld.verticalWall(2, 3, 1);
		gridWorld.verticalWall(3, 3, 4);
		
		//gridWorld.generatePrimMap(numLocations);
		gridWorld.setProbSucceedTransitionDynamics(0.5);
		//gridWorld.setDeterministicTransitionDynamics();
		domain = gridWorld.generateDomain();

		// create the state parser
		sp = new GridWorldStateParser(domain);

		ArrayList<Point> endPoints = new ArrayList<Point>(7);
		endPoints.add(new Point(0, 0, -10.));
		endPoints.add(new Point(1, 0, -10.));
		endPoints.add(new Point(2, 0, -10.));
		endPoints.add(new Point(3, 0, -10.));
		endPoints.add(new Point(4, 0, -10.));
		endPoints.add(new Point(2, 2, -1.));
		endPoints.add(new Point(4, 2, 10.));
		gridWorld.setEndPoints(endPoints);
		
		rf = new PointRewarder(endPoints, defaultCost);
		tf = new SinglePFTF(domain.getPropFunction(GridWorldDomain.PFATLOCATION));
		goalCondition = new TFGoalCondition(tf);

		// set up the initial state of the task
		initialState = PrimGridWorldDomain.getOneAgentNLocationState(domain, endPoints.size());
		PrimGridWorldDomain.setAgent(initialState, 1, 1);
		for (int i = 0; i < endPoints.size(); i++) {
			final Point ep = gridWorld.getEndPoints().get(i);
			PrimGridWorldDomain.setLocation(initialState, i, ep.x, ep.y);
		}
		// set up the state hashing system
		hashingFactory = new DiscreteStateHashFactory();
		hashingFactory.setAttributesForClass(GridWorldDomain.CLASSAGENT, domain.getObjectClass(GridWorldDomain.CLASSAGENT).attributeList);

		stateSpaceObserver = new StateSpaceObserver(hashingFactory, initialState);
		((SADomain) this.domain).setActionObserverForAllAction(stateSpaceObserver);

		 // add visual observer
//		 VisualActionObserver observer = new VisualActionObserver(domain,
//		 GridWorldVisualizer.getVisualizer(domain, gridWorld.getMap()));
//		 ((SADomain) this.domain).setActionObserverForAllAction(observer);
//		 observer.initGUI();

	}

	public AnalysisData runValueIteration(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}

		ValueIteration planner = new ValueIteration(domain, rf, tf, 0.99, hashingFactory, 0.001, 1000);
		planner.planFromState(initialState);

		// create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner) planner);

		// record the plan results to a file
		long start = System.currentTimeMillis();
		EpisodeAnalysis ea = p.evaluateBehavior(initialState, rf, tf);
		long finish = System.currentTimeMillis();
		AnalysisData data = generateAnalysisData(ea);
		data.elapsed = finish-start;
		data.iterationsUntilConversion = planner.getIterationsUntilConvergence();
		return data;
	}

	public AnalysisData runPolicyIteration(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}

		PolicyIteration planner = new PolicyIteration(domain, rf, tf, 0.99, hashingFactory, 0.001, 1000, 2000);
		planner.planFromState(initialState);

		// create a Q-greedy policy from the planner
		Policy p = new GreedyQPolicy((QComputablePlanner) planner);

		// record the plan results to a file
		long start = System.currentTimeMillis();
		EpisodeAnalysis ea = p.evaluateBehavior(initialState, rf, tf);
		long finish = System.currentTimeMillis();
		
		AnalysisData data = generateAnalysisData(ea);
		data.elapsed = finish-start;
		data.iterationsUntilConversion = planner.getIterationsUntilConvergence();
		return data;
	}

	private AnalysisData generateAnalysisData(EpisodeAnalysis ea) {
		double reward = ea.getDiscountedReturn(1);
		State last = ea.stateSequence.get(ea.stateSequence.size() - 1);
		ObjectInstance inst = last.getFirstObjectOfClass("agent");
		int x = inst.getDiscValForAttribute("x");
		int y = inst.getDiscValForAttribute("y");
		Point ep = gridWorld.getEndPoints().get(gridWorld.getEndPoints().size() - 1);
		boolean isMax = ep.x == x && ep.y == y;
		// if (isMax) {
		// System.out.println("Found max reward");
		// } else {
		// System.out.println(String.format("Max:%d,%d,pt=%s", x, y, ep));
		// }
		System.out.println(ea.getActionSequenceString());
		System.out.println(String.format("Returned reward: %.2f, # states: %d", reward, ea.stateSequence.size()));
		return new AnalysisData(stateSpaceObserver.getTotalStateVisits(), stateSpaceObserver.getNumUniqueVisitedStates(),
				ea.stateSequence.size(), reward, isMax ? 1 : 0);
	}

	public AnalysisData runQLearning(String outputPath) {

		if (!outputPath.endsWith("/")) {
			outputPath = outputPath + "/";
		}

		// discount= 0.99; initialQ=0.0; learning rate=0.9
		QLearning agent = new QLearning(domain, rf, tf, 0.99, hashingFactory, 0., 0.9);
		agent.setNumEpisodesToStore(100);
		agent.setMaximumEpisodesForPlanning(1000);
		// run learning for 100 episodes
//		EpisodeAnalysis ea = null;
		long start = System.currentTimeMillis();
//		for (int i = 0; i < 100; i++) {
//			ea = agent.runLearningEpisodeFrom(initialState);
//			// ea.writeToFile(String.format("%se%03d", outputPath, i), sp);
//			// System.out.println(i + ": " + ea.numTimeSteps());
//		}
		agent.planFromState(initialState);
		
		long finish = System.currentTimeMillis();

		AnalysisData data = generateAnalysisData(agent.getLastLearningEpisode());
		data.elapsed = finish-start;
		return data;

	}

	public void valueFunctionVisualize(QComputablePlanner planner, Policy p) {
		List<State> allStates = StateReachability.getReachableStates(initialState, (SADomain) domain, hashingFactory);
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);

		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX, GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphRenderStyle.DISTSCALED);

		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, planner);
		gui.setSpp(spp);
		gui.setPolicy(p);
		gui.setBgColor(Color.GRAY);
		gui.initGUI();
	}

	class PointRewarder implements RewardFunction {

		private List<Point> points;
		private double defaultReward;

		PointRewarder(List<Point> points, double reward) {
			this.points = points;
			defaultReward = reward;
		}

		@Override
		public double reward(State s, GroundedAction a, State sprime) {
			ObjectInstance inst = sprime.getFirstObjectOfClass("agent");
			int x = inst.getDiscValForAttribute("x");
			int y = inst.getDiscValForAttribute("y");

			// see if we moved to point that has a specific reward
			for (Point p : points) {
				if (p.x == x && p.y == y) {
					return p.reward;
				}
			}
			return defaultReward;
		}

	}

	class StateSpaceObserver implements ActionObserver {

		private StateHashFactory hashFactory;
		private Map<StateHashTuple, Integer> stateSpaceMap = new HashMap<StateHashTuple, Integer>();
		private int totalStateVisits = 0;

		public StateSpaceObserver(StateHashFactory hashFactory, State initial) {
			this.hashFactory = hashFactory;
			countState(initial);
		}

		public int getNumUniqueVisitedStates() {
			return stateSpaceMap.size();
		}

		private void countState(State st) {
			// get the hash
			StateHashTuple hash = hashFactory.hashState(st);
			// see if we've been here before
			Integer count = stateSpaceMap.get(hash);
			if (count == null) {
				// first visit
				count = 1;
			} else {
				// been here before - increment
				count++;
			}

			stateSpaceMap.put(hash, count);
			totalStateVisits++;
		}

		public int getTotalStateVisits() {
			return totalStateVisits;
		}

		@Override
		public void actionEvent(State s, GroundedAction ga, State sp) {
			// ObjectInstance o = sp.getFirstObjectOfClass("agent");
			// int x = o.getDiscValForAttribute("x");
			// int y = o.getDiscValForAttribute("y");
			// System.out.println(String.format("visit %d,%d", x, y));
			countState(sp);
		}

	}

	static class AnalysisData {
		int totalStateVisits;
		int uniqueStates;
		int totalStates;
		double reward;
		int maxRewardCount = 0;
		long elapsed = 0;
		long iterationsUntilConversion;

		AnalysisData(int totalVisits, int unique, int total, double reward, int maxRewardCount) {
			totalStateVisits = totalVisits;
			uniqueStates = unique;
			totalStates = total;
			this.reward = reward;
			this.maxRewardCount = maxRewardCount;
		}
	}

}