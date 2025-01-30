
## Publications, Academic Projects and Thesis 

In this repository, I showcase my academic journey during my Master's degree in Computer Science. It includes my **thesis**, **publications**, and various **projects**.
 The material spans across multiple areas of computer science such as **algorithms**, **automata theory**, **logic** and **parallel computing**.

### Publications
<table>



<tr> 
     <td><a href="https://doi.org/10.1002/wcms.1691"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/web.png" width="50"><br> DOI</a></td>
	 <td></td>
	<td>
	 <details>
	 <summary><a href="https://doi.org/10.1002/wcms.1691">The kth nearest neighbor method for estimation of entropy changes from molecular ensembles</a></summary>
	 <b>Type: </b> Publication<br>
	 <b>Language: </b> English<br>
	 <b>Authors: </b> Federico Fogolari, Roberto Borelli, Agostino Dovier, Gennaro Esposito<br>
	 <b>Abstract: </b> All processes involving molecular systems entail a balance between associated enthalpic and entropic changes. Molecular dynamics simulations of the end-points of a process provide in a straightforward way the enthalpy as an ensemble average. Obtaining absolute entropies is still an open problem and most commonly pathway methods are used to obtain free energy changes and thereafter entropy changes. The kth nearest neighbor (kNN) method has been first proposed as a general method for entropy estimation in the mathematical community 20 years ago. Later, it has been applied to compute conformational, positional–orientational, and hydration entropies of molecules. Programs to compute entropies from molecular ensembles, for example, from molecular dynamics (MD) trajectories, based on the kNN method, are currently available. The kNN method has distinct advantages over traditional methods, namely that it is possible to address high-dimensional spaces, impossible to treat without loss of resolution or drastic approximations with, for example, histogram-based methods. Application of the method requires understanding the features of: the kth nearest neighbor method for entropy estimation; the variables relevant to biomolecular and in general molecular processes; the metrics associated with such variables; the practical implementation of the method, including requirements and limitations intrinsic to the method; and the applications for conformational, position/orientation and solvation entropy. Coupling the method with general approximations for the multivariable entropy based on mutual information, it is possible to address high dimensional problems like those involving the conformation of proteins, nucleic acids, binding of molecules and hydration.
	 </details>
	 </td>
</tr>

<tr> 
     <td><a href="https://doi.org/10.3390/biophysica2040031"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/web.png" width="50"><br> DOI</a></td>
	 <td><a href="https://github.com/robyBorelli/nearest-neighbours-package"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/code.png" width="50"><br> Code</a></td>
	<td>
	 <details>
	 <summary><a href="https://doi.org/10.3390/biophysica2040031">Data Structures and Algorithms for k-th Nearest Neighbours Conformational Entropy Estimation</a></summary>
	 <b>Type: </b> Publication<br>
	 <b>Language: </b> English<br>
	 <b>Authors: </b> Roberto Borelli, Agostino Dovier and Federico Fogolari<br>
	 <b>Abstract: </b> Entropy of multivariate distributions may be estimated based on the distances of nearest neighbours from each sample from a statistical ensemble. This technique has been applied on biomolecular systems for estimating both conformational and translational/rotational entropy. The degrees of freedom which mostly define conformational entropy are torsion angles with their periodicity. In this work, tree structures and algorithms to quickly generate lists of nearest neighbours for periodic and non-periodic data are reviewed and applied to biomolecular conformations as described by torsion angles. The effect of dimensionality, number of samples, and number of neighbours on the computational time is assessed. The main conclusion is that using proper data structures and algorithms can greatly reduce the complexity of nearest neighbours lists generation, which is the bottleneck step in nearest neighbours entropy estimation.
	 </details>
	 </td>
</tr>
</table>

---

### Academic Projects
<table>

<tr> 
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/expressiveness_of_transformers.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	 <td>
	 <td></td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/expressiveness_of_transformers.pdf">On The Expressiveness Of Masked Hard-Attention Transformers</a></summary>
	 <b>Type: </b> Seminar for the course "Foundations of Neural Networks"<br>
	 <b>Language: </b> English<br>
	 <b>Abstract: </b>  This work presents the paper by Yang et al. which characterizes the expressiveness of a particular class of transformers with hard attention, where attention is focused on exactly one position at a time. It is shown how to compile a transformer model into the language B-RASP. Furthermore, it is established that B-RASP is equivalent to star-free languages. The proof proceeds in two directions, employing two distinct characterizations of star-free languages: linear temporal logic over finite traces and cascades of reset automata. This study offers a deeper understanding of the transformer formalism, revealing that (i) both the feed-forward and self-attention sublayers play crucial roles, and (ii) increasing the number of layers in a transformer enhances its expressive power. This latter result contrasts with the universal approximation theorem for standard feedforward neural networks.
	 </details>
	 </td>
</tr>

<tr> 
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/3sum.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Report</a></td>
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/3sum.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/3sum.pdf">The 3SUM problem admits subquadratic solutions</a></summary>
	 <b>Type: </b> Seminar for the course "Advanced Algorithms"<br>
	 <b>Language: </b> English<br>
	 <b>Abstract: </b> In this work, I consider the 3sum problem. Recent years’ studies have shown that the problem admits a subquadratic solution. The 3sum problem has been used in the area of fine-grained complexity to establish lower bounds to a wide range of other problems (which have shown to be 3sum-hard) for example in the computational geometry area. In this paper, I examine the Freund approach to obtain a subquadratic algorithm. To obtain a saving in the complexity, several tricks have been applied and in particular it has been shown how to efficiently enumerate the so-called chunks through a correspondence with paths in a matrix and then all pairs of blocks agreeing with such derived chunks are obtained through a reduction to the dominance-merge problem.
	 </details>
	 </td>
</tr>

<tr> 
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/expressiveness_of_temporal_logic.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Report</a></td>
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/expressiveness_of_temporal_logic.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/expressiveness_of_temporal_logic.pdf">Expressiveness of temporal logic</a></summary>
	 <b>Type: </b> Seminar for the course "Automatic system Verification: Theory and Applications"<br>
	 <b>Language: </b> English<br>
	 <b>Abstract: </b> In this work, I consider the expressive power of various temporal logics. First, I recall some basic results about expressiveness of first order logic. Then I consider the case of LTL and I show a theorem that can be used to prove that the concept of parity is not definable in this context. I discuss a counterexample that proves that the mentioned theorem doesn’t directly apply to LTL+P and I briefly highlight how a possible investigation may lead to a generalization of the theorem to the LTL+P case. Next, I relate first order definable languages with LTL ones and I present an extension to LTL which allows us to increase the expressive power and capture regular languages without changing the complexity of the decision procedure. Finally, I move to the more interesting case of interval logic. I introduce the notion of bisimulation and its use in modal logic and, in particular, I show how to apply it to prove that the logic PNL is strictly more expressive than its future fragment A.
	 </details>
	 </td>
</tr>

<tr> 
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/logic_for_cf_languages.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	 <td></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/logic_for_cf_languages.pdf">Logic for context-free languages</a></summary>
	 <b>Type: </b> Seminar for the course "Logic for Computer Science"<br>
	 <b>Language: </b> Italian<br>
	 <b>Abstract: </b> 
	 </details>
	 </td>
</tr>


<tr> 
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/automata_minimization.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Report</a></td>
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/code/automata_minimization"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/code.png" width="50"><br> Code</a></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/automata_minimization.pdf">Parallel Automata Minimization</a></summary>
	 <b>Type: </b> Project for the course "Programming on Parallel Architectures"<br>
	 <b>Language: </b> Italian<br>
	  <b>Authors: </b> Roberto Borelli (OpenMP), Stefano Rocco (CUDA)<br>
	 <b>Abstract: </b> The minimization problem of an automaton is central in automata theory and has various practical implications. In this work, we aim to develop a parallel version of the well-known Moore's algorithm, which classically runs in quadratic time. We will review the fundamental concepts and problems in the field, analyze the serial algorithm by examining its code, theoretical properties, and time complexity. Using the OpenMP programming model, we will develop six different parallel versions of the algorithm. The first four, more efficient versions, are based on dividing the main loop into parallel tasks. The fifth version addresses the issue of merging multiple iterations of the refinement loop. The sixth and most scalable version will attempt an approach based on the parallelization of RadixSort and, ultimately, CountingSort, which will then be further developed in CUDA. We will divide CountingSort into three phases, proposing various implementation solutions for each phase using this programming model. We will test and compare the OpenMP and CUDA implementations on a significant set of instances.
	 </details>
	 </td>
</tr>

<tr> 
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/scheduling_competitions_ar.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Report</a></td>
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/code/scheduling_competitions_ar"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/code.png" width="50"><br> Code</a></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/scheduling_competitions_ar.pdf">Scheduling of competitions with automated reasoning techniques</a></summary>
	 <b>Type: </b> Final project for the course "Automated Reasoning"<br>
	 <b>Language: </b> Italian<br>
	 <b>Abstract: </b> 
	 </details>
	 </td>
</tr>

<tr> 
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/xgboost.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Report</a></td>
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/xgboost.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	 <td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/xgboost.pdf">XGBoost</a></summary>
	 <b>Type: </b> Seminar for the course "Applied Statistics and Data Analysis"<br>
	 <b>Language: </b> Italian<br>
	 <b>Abstract: </b> 
	 </details>
	 </td>
</tr>


</table> 

---


### Thesis
<table>
<tr> 
	 <td><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/thesis.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/pdf.png" width="50"><br> Thesis</a></td>
     <td><a href="https://github.com/robyBorelli/Seminars/blob/main/presentations/thesis.pdf"><img src="https://raw.githubusercontent.com/robyBorelli/Seminars/main/images/slides.png" width="50"><br> Slides</a></td>
	<td>
	 <details>
	 <summary><a href="https://github.com/robyBorelli/Seminars/blob/main/reports/thesis.pdf">Bachelor's thesis: Algorithms for neighbourhood searches</a></summary>
	 <b>Type: </b> Bachelor's thesis<br>
	 <b>Language: </b> Italian<br>
	 <b>Abstract: </b> 
	 </details>
	 </td>
</tr>
</table>

---

