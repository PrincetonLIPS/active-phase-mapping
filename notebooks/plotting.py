# Plotting utilities
import matplotlib.pyplot as plt
import jax.numpy as jnp 
from active_search import get_next_y
import ternary


def plot_candidate(knot_x, true_y, true_envelope, pred_mean, pred_cov, envelopes, dataset, next_x, scores, plot_eig=True, legend=True, plot_hulls=True):
    """
    Plotting utilities for the 1D example.
    """
    
    plt.figure(figsize=(10,4))

    # Plot true function, convex hull, and GP model mean + uncertainty
    plt.plot(knot_x, true_y, "k", lw=2, label="True function")
    plt.plot(knot_x, true_envelope.T, ls="dashed", label="True envelope", lw=2, c="k")
    plt.plot(knot_x, pred_mean, lw=2, c="tab:blue", label="Model prediction")
    y_err = 2 * jnp.sqrt(jnp.diag(pred_cov))
    plt.fill_between(knot_x, pred_mean - y_err, pred_mean + y_err, alpha=0.4, color="tab:blue")
    
    if plot_hulls:
        # Plot convex hulls of posterior samples
        plt.plot(knot_x, envelopes[:15,:].T, lw=0.5, c="gray")

    # Plot data / next evaluation
    plt.scatter(next_x, get_next_y(true_y, knot_x[:,jnp.newaxis], next_x), marker="*", color="tab:red", zorder=5, sizes=[150], label="Next evaluation")
    plt.scatter(dataset.X, dataset.y, label="Observed data", c="k", zorder=5)

    # Plot the entropy estimates (i.e., -second term of EIG)
    if plot_eig:
        plt.scatter(knot_x, scores, c="purple", marker="|", label="EIG")
        
    if legend:
        plt.legend(ncol=2)

    plt.ylim(-3,3); plt.xlabel("Composition space"); plt.ylabel("Energy")
    
    
def plot_eig(knot_x, scores):
    plt.figure(figsize=(10,4))
    plt.scatter(knot_x, scores, c="purple", marker=".", label="EIG")
    #if legend:
    #    plt.legend(ncol=2)

    plt.ylim(-3,3); plt.xlabel("Composition space"); plt.ylabel("Energy")
    


def plot_triangle(pts, tight_pts, data, next_x = None, plot_design = True):
    fontsize=20
    ### Scatter Plot
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    #tax.set_title("", fontsize=20)
    tax.boundary(linewidth=2.0)
    #tax.gridlines(multiple=1, color="blue")
    
    x_train_pts = []
    for x in data[0]:
        x_train_pts.append((x[0], x[1], 1-x.sum()))

    if plot_design:
        tax.scatter(pts, marker=".", label="Design space",  sizes=jnp.ones(len(pts))*150, c="gray", zorder=3)
    tax.scatter(x_train_pts, marker=".", label="Observation", sizes=jnp.ones(len(pts))*250, zorder=3, c="k")
    tax.scatter(tight_pts, marker="*", label="Tight points", sizes=jnp.ones(len(pts))*250, zorder=5, c="tab:blue", alpha=0.5)
    
    if next_x != None:
        tax.scatter([pts[next_x]], marker="*", label="Candidate", sizes=jnp.ones(len(pts))*250, zorder=6, c="tab:red")

    tax.legend(loc="upper right")
    tax.right_corner_label("B", fontsize=fontsize)
    tax.top_corner_label("A", fontsize=fontsize)
    tax.left_corner_label("C", fontsize=fontsize)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=1, multiple=1, offset=0.02, fontsize=12)
    tax.show()
    
    
