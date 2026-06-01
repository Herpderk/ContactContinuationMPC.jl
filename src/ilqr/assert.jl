function assert_opts!(opts::iLQROptions)::Nothing
    if !(0.0 < opts.alpha_mul < 1.0)
        Utils.throwdom(
            opts.alpha_mul,
            "The backtracking contraction rate must be between 0 and 1",
        )
    end
    if opts.eps_reg <= 0.0
        Utils.throwdom(
            opts.eps_reg,
            "The regularizer coefficient must be greater than or equal to 0",
        )
    end
    if opts.eps_fd < 0.0
        Utils.throwdom(
            opts.eps_fd,
            "The finite-difference coefficient must be greater than 0",
        )
    end
    if opts.tol_interp < 0.0
        Utils.throwdom(
            opts.tol_interp,
            "The dynamics jacobian interpolation tolerance must be greater than 0",
        )
    end
    if opts.tol_constr < 0.0
        Utils.throwdom(
            opts.tol_constr, "The constraint tolerance must be greater than 0"
        )
    end
    if opts.tol_ilqr < 0.0
        Utils.throwdom(
            opts.tol_ilqr,
            "The iLQR convergence tolerance must be greater than 0",
        )
    end
    if opts.margin_ls < 0.0
        Utils.throwdom(
            opts.margin_ls,
            "The merit function margin factor must be greater than or equal to 0",
        )
    end
    if opts.maxiter_al < 0
        Utils.throwdom(
            opts.maxiter_al,
            "The max number of outer AL iterations must be greater than 0",
        )
    end
    if opts.maxiter_ilqr < 0
        Utils.throwdom(
            opts.maxiter_ilqr,
            "The max number of inner iLQR iterations must be greater than 0",
        )
    end
    if opts.maxiter_ls < 0
        Utils.throwdom(
            opts.maxiter_ls,
            "The max number of line-search iterations must be greater than 0",
        )
    end
    return nothing
end
