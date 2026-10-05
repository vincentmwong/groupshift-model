# Design discussion

The purpose of this markdown file is to operate as a working document for discussing pending project changes and ideas. This workspace can be used to house ideas that are broader or larger in scale and scope than line-item notes inside of the manuscript itself. This is a living document. 



## Changes to the project narrative

VW: I want to make a major change to the overarching narrative of the project. 

I believe that the main mechanism of this paper is this: 
Although these two mechanisms produce system-wide group sorting independently, combining the two mechanisms causes the system to fail to sort as clearly. This occurs because the two mechanisms are actually antagonists, not agonists, to group polarization. This leads to multiple conflicting configurations where ingroup perceptions are more accurate than expected. 

I think I can demonstrate this in the following steps: 
* Show that the Attraction mechanism produces "optimal group sorting"/"maximal polarization" on its own. 
    * The expected subfigure will be a plot of average group perceptions and average group opinions over time, across many simulations. Based on simulations not yet shown here, there will be a robust separation of perceptions and thus a separation/sorting of groups across simulations. I have always run this sanity check but I realized I needed to demonstrate it here. 
    * This will show that the drifting effect of the Attraction mechanism is sufficient to cause minor perturbations in the initial system to compound into a separation of groups. This matches the expected behavior of other ABMs of conformity and group separation/polarization. 
    * This will also show that the Attraction mechanism produces systems where groups are sorted and separated while also being accurate about their perceptions. This is based on my experience running such simulations. 
* Show that the Repulsion mechanism ALSO produces optimal group sorting on its own. 
    * The expected subfigure will also be a plot of average group perceptions over time. This will show a robust separation of perceptions and thus groups, however the perceptions will continue drifting away from the midpoint of the opinion space. This is because, unlike Attraction, the Repulsion mechanic's calculation only accounts for other group perceptions, and the size of the effect asymptotically approaches 0, so there's always a nonzero drift. 
    * This will show that the Repulsion effect is also sufficient to 
* Lastly, show that the Attraction+Repulsion models produce group sorting failure. 
    * This is represented in the simulations currently shown in the paper. Groups do not sort nearly as readily as expected, and group perceptions are not as clear as expected. 
    * The strongest example of this occurs when Repulsion parameters are high. That parameter represents people's tendnecy to perceive group polarization, which SHOULD produce very clear examples of false polarization. 
    * This could even be shown in a figure where high Repulsion on its own produces significant false polarization, but adding even a small amount of Attraction causes that to break down. 


After that is established, the paper can continue on to show the investigation of the system's response to changes in parameters. These sections are currently in the paper. This will show that when we test how sensitive this system is to changes in parameters, we find that the system is actually MORE robust than expected. Increasing the tendency to see groups as different or increasing the tendency to adjust perceptions to new group information actually does surprisingly little to help the group sort (contradicting intuitions), and it also causes an unintuitively small difference in the expected amount of group misperceptions. 


I want to then change the paper introduction. First, I want to introduce pluralistic ignorance as a problem. Then, I want to introduce that seemingly competing mechanisms have been shown to produce the same group sorting issues in both verbal models used in social psychology AND in ABMs of group dynamics. (unsure about this part, I think I need to be very careful about whether mechanisms are proposed to produce configurational group polarization or group misperceptions)
