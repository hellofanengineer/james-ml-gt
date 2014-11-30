package org.assign4;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.State;
import burlap.oomdp.singleagent.explorer.TerminalExplorer;
import burlap.oomdp.singleagent.explorer.VisualExplorer;
import burlap.oomdp.visualizer.Visualizer;

/**
 * A random Prim gridworld generator based on code from http://jonathanzong.com/blog/2012/11/06/maze-generation-with-prims-algorithm.
 * 
 * The code has been adapted to support multiple terminal points with varying rewards.
 * 
 * @author james
 *
 */
public class PrimGridWorldDomain extends GridWorldDomain implements DomainGenerator {

	private Point agentStartPoint;
	private Point endPoint;

	private List<Point> endPoints;

	private Random rand;

	public PrimGridWorldDomain(int width, int height) {
		super(width, height);
		rand = RandomFactory.getMapped(0);
	}
	
	public int countStates() {
		int count = 0;
		
		if (map != null && map.length > 0) {
			for (int r=0; r < map.length; r++) {
				int[] row = map[r];
				for (int i=0; i < row.length; i++) {
					if (row[i] == 0) {
						// reachable state
						count++;
					}
				}
			}
		}
		
		return count;
	}

	public void generatePrimMap(int numEndPoints) {
		int[][] map = new int[width][height];
		// init to all walls
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				map[x][y] = 1;
			}
		}

		// int x, y;
		// select random point and open as start node
		Point st = new Point((int) (rand.nextDouble() * width), (int) (rand.nextDouble() * height), null);
		Point startPoint = st;
		Point endPoint = null;
		// map[st.x][st.y] = 2;
		map[st.x][st.y] = 0;

		// iterate through direct neighbors of node
		ArrayList<Point> frontier = new ArrayList<Point>();
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				if (x == 0 && y == 0 || x != 0 && y != 0)
					continue;
				try {
					if (map[st.x + x][st.y + y] == 0)
						continue;
				} catch (Exception e) { // ignore ArrayIndexOutOfBounds
					continue;
				}
				// add eligible points to frontier
				frontier.add(new Point(st.x + x, st.y + y, st));
			}
		}

		Point last = null;
		while (!frontier.isEmpty()) {

			// pick current node at random
			Point cu = frontier.remove((int) Math.round((rand.nextDouble() * (double) (frontier.size() - 1))));
			Point op = cu.opposite();
			try {
				// if both node and its opposite are walls
				if (map[cu.x][cu.y] == 1) {
					if (map[op.x][op.y] == 1) {

						// open path between the nodes
						map[cu.x][cu.y] = 0;
						map[op.x][op.y] = 0;

						// store last node in order to mark it later
						last = op;

						// iterate through direct neighbors of node, same as
						// earlier
						for (int x = -1; x <= 1; x++)
							for (int y = -1; y <= 1; y++) {
								if (x == 0 && y == 0 || x != 0 && y != 0)
									continue;
								try {
									if (map[op.x + x][op.y + y] == 0)
										continue;
								} catch (Exception e) {
									continue;
								}
								frontier.add(new Point(op.x + x, op.y + y, op));
							}
					}
				}
			} catch (Exception e) { // ignore NullPointer and
									// ArrayIndexOutOfBounds
			}

			// if algorithm has resolved, mark end node
			if (frontier.isEmpty()) {
				map[last.x][last.y] = 0;
				endPoint = new Point(last.x, last.y, null);
			}
		}

		// print final maze - want 0 to be at bottom so print rows from the end
		// first
//		for (int i = map.length - 1; i >= 0; i--) {
//			for (int j = 0; j < map[i].length; j++) {
//				System.out.print(map[i][j]);
//			}
//			System.out.println();
//		}
//		System.out.println("start=" + startPoint + ";end=" + endPoint);

		this.agentStartPoint = startPoint;
		this.endPoint = endPoint;
		
		endPoints = new ArrayList<Point>(numEndPoints);
		endPoints.add(endPoint);
		
		// add additional points if necessary
		if (numEndPoints > 1) {
			int remainingPoints = numEndPoints - 1;
			while (remainingPoints > 0) {
				int randX = generateRandomCoordinate(width);
				int randY = generateRandomCoordinate(height);
				// look for a random wall point that is also accessible to a path
				boolean accessible = (randY < height-1 && map[randX][randY+1] == 0) || 
						(randY > 0 && map[randX][randY-1] == 0) || 
						(randX < width-1 && map[randX+1][randY] == 0) || 
						(randX > 0 && map[randX-1][randY] == 0);
				if (accessible && map[randX][randY] == 1) {
					// change the wall to a point
					map[randX][randY] = 0;
					Point ep = new Point(randX, randY, null);
					System.out.println("end point=" + ep);
					endPoints.add(ep);
					remainingPoints--;
				}
			}
		}

		super.setMap(map);
	}

	private int generateRandomCoordinate(int max) {
		return (int) Math.round(rand.nextDouble() * (double) (max-1));
	}

	public Point getAgentStartPoint() {
		return agentStartPoint;
	}

	public Point getEndPoint() {
		return endPoint;
	}
	
	public List<Point> getEndPoints() {
		return endPoints;
	}

	public void setAgentStartPoint(Point agentStartPoint) {
		this.agentStartPoint = agentStartPoint;
	}

	public void setEndPoints(List<Point> endPoints) {
		this.endPoints = endPoints;
	}

	static class Point {
		Integer x;
		Integer y;
		Point parent;
		double reward = -1.;

		public Point(int x, int y, Point p) {
			this.x = x;
			this.y = y;
			parent = p;
		}
		
		public Point(int x, int y, double reward) {
			this.x = x;
			this.y = y;
			this.reward = reward;
		}

		// compute opposite node given that it is in the other direction from
		// the parent
		public Point opposite() {
			if (this.x.compareTo(parent.x) != 0)
				return new Point(this.x + this.x.compareTo(parent.x), this.y, this);
			if (this.y.compareTo(parent.y) != 0)
				return new Point(this.x, this.y + this.y.compareTo(parent.y), this);
			return null;
		}

		public String toString() {
			return x.toString() + ',' + y.toString();
		}
	}

	public static void main(String[] args) {
		PrimGridWorldDomain prim = new PrimGridWorldDomain(11, 11);

		int numLocations = 4;
		
		prim.generatePrimMap(numLocations);
		prim.setProbSucceedTransitionDynamics(0.8);

		Domain d = prim.generateDomain();
		State s = getOneAgentNLocationState(d, numLocations);
		setAgent(s, prim.getAgentStartPoint().x, prim.getAgentStartPoint().y);
		for (int i=0; i < numLocations; i++) {
			final Point ep = prim.getEndPoints().get(i);
			setLocation(s, i, ep.x, ep.y);
		}
		
		int expMode = 1;
		if (args.length > 0) {
			if (args[0].equals("v")) {
				expMode = 1;
			} else if (args[0].equals("t")) {
				expMode = 0;
			}
		}

		if (expMode == 0) {

			TerminalExplorer exp = new TerminalExplorer(d);
			exp.addActionShortHand("n", ACTIONNORTH);
			exp.addActionShortHand("e", ACTIONEAST);
			exp.addActionShortHand("w", ACTIONWEST);
			exp.addActionShortHand("s", ACTIONSOUTH);

			exp.exploreFromState(s);

		} else if (expMode == 1) {

			Visualizer v = GridWorldVisualizer.getVisualizer(prim.getMap());
			VisualExplorer exp = new VisualExplorer(d, v, s);

			// use w-s-a-d-x
			exp.addKeyAction("w", ACTIONNORTH);
			exp.addKeyAction("s", ACTIONSOUTH);
			exp.addKeyAction("a", ACTIONWEST);
			exp.addKeyAction("d", ACTIONEAST);

			exp.initGUI();
		}

	}
}
