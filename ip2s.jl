
using Nemo
using AbstractAlgebra
# Additions to Nemo module<<<1
# Nemo.nmod fixes<<<2
#
# fix (Z/3)(Mod(1,6)) so that it lives in Z/3 instead of Z/6.
@inline (R::Nemo.NmodRing)(a::Nemo.nmod) = _coerce_nmod(Val(parent(a)==R), R, a)
@inline _coerce_nmod(::Val{true},  R::Nemo.NmodRing, a::Nemo.nmod)= a
@inline _coerce_nmod(::Val{false}, R::Nemo.NmodRing, a::Nemo.nmod)= R(a.data)

# enrich `show` to display full info for nmod values
@inline show_mod(io::IO, a::Nemo.nmod) =
	print(io, "Mod(", signed(widen(a.data)), ",", signed(widen(parent(a).n)),")")
@inline Base.show(io::IO, ::MIME"text/plain", a::Nemo.nmod) = show_mod(io, a)

# a convenient `Mod` constructor for nmods
@inline Mod(a::Integer, b::Integer) = Nemo.nmod(a, Nemo.ResidueRing(Nemo.ZZ, b))

# Allow symbols as variables <<<2
# interface:
# allowing symbols instead of strings for variables
@inline Nemo.PolynomialRing(R::Nemo.Ring, S::Symbol; cached::Bool=true) =
	Nemo.PolynomialRing(R, string(S), cached=cached)
@inline Nemo.FlintFiniteField(p::Union{fmpz,Integer}, f::Int,
	s::Symbol; cached::Bool=true) =
	Nemo.FlintFiniteField(p, f, String(s), cached=cached)
@inline Nemo.FlintFiniteField(p::Union{fmpz,Integer}; cached::Bool=true) =
	Nemo.FlintFiniteField(p, 1, "g", cached=cached)[1]
@inline Nemo.PowerSeriesRing(R::FlintIntegerRing, prec::Int, s::Symbol;
	cached=true) = Nemo.PowerSeriesRing(R, prec, String(s), cached=cached)


# Operators on sets (quotient rings, free modules) <<<2
# \mathbb notations
ℤ=Nemo.ZZ
ℚ=Nemo.QQ
FF=Nemo.FiniteField
NullRing=Nemo.ResidueRing(Nemo.ZZ,1)

# ℤ/2 = quotient ring
# ℤ^2 = free module
# ℤ^(2,2) = matrix ring
# TODO: module / submodule
@inline Base.:/(R::Nemo.AbstractAlgebra.Ring, n::Integer) =
	Nemo.ResidueRing(R, n)
@inline Base.:^(R::Nemo.AbstractAlgebra.Ring, n::Integer) =
	Nemo.FreeModule(R, n)
@inline Base.:^(R::Nemo.AbstractAlgebra.Ring, n::NTuple{2,<:Integer}) =
	Nemo.MatrixSpace(R, n[1], n[2])

# ℤ[:x] = polynomial ring
# ℤ[[:x]] = formal series
# ℚ(:x) = fraction field
# TODO: allow same for multivariate
@inline Base.getindex(R::Nemo.AbstractAlgebra.Ring, S::Symbol) =
	Nemo.PolynomialRing(R, S)
# FIXME make power series precision parametrizable
@inline Base.getindex(R::Nemo.AbstractAlgebra.Ring, a::Vector{Symbol}) = begin
	@assert length(a) == 1
	Nemo.PowerSeriesRing(R, 16, String(a[1]))
end
@inline (R::Nemo.AbstractAlgebra.Field)(S::Symbol) =
	Nemo.FractionField(Nemo.PolynomialRing(R, S)[1])

# Iterators on matrices<<<2
function Base.iterate(a::fq_nmod_mat, state=(1,1))
	(state[2] > ncols(a)) && return nothing
	newstate = (state[1] >= nrows(a)) ? (1, state[2]+1) : (state[1]+1, state[2])
	return (a[state...], newstate)
end
@inline Base.IteratorSize(::Nemo.fq_nmod_mat) = Base.HasShape{2}()


# Amalgamation <<<1
# Infrastructure<<<2
"""
    amalg_rule(op, parent1, parent2)

Says what is the parent object of operations `op(x1, x2)` for `x1 ∈
parent1` and `x2 ∈ parent2`. Like `promote_rule`, this function should
not be called directly.
"""
@inline amalg_rule(op, p1, p2) = nothing

"""
    amalg_parent(op, parent1, parent2)
Says what is the parent object of operations `op(x1,x2)` for `x1 ∈
parent1` and `x2 ∈ parent2`.
"""
@inline function amalg_parent(op, p1, p2)
	return amalg_result(op, Val(p1==p2),
		p1, p2, amalg_rule(op, p1, p2), amalg_rule(op, p2, p1))
end
# same parent: return given parent
# FIXME: restrict to allowed operations?
amalg_result(op, ::Val{true}, p1, p2, r12, r21) = p1
amalg_result(op, ::Val{false}, p1, p2, ::Nothing, ::Nothing) = nothing
amalg_result(op, ::Val{false}, p1, p2, r, s) = r
amalg_result(op, ::Val{false}, p1, p2, ::Nothing, r) = r

# TODO: Groups<<<2
AnyGroupOp = Union{typeof(+), typeof(-)}
# Rings<<<2
AnyRingOp = Union{AnyGroupOp, typeof(*)}
Base.show(io::IO, ::Type{AnyRingOp})=print(io, "AnyRingOp")
macro amalg_ring(r1, r2)
	:(amalg_rule(::AnyRingOp, $(esc(r1)), $(esc(r2))))
end
# disambiguation:
@amalg_ring(::FlintIntegerRing, Z::FlintIntegerRing) = Z
@amalg_ring(::FlintIntegerRing, R::AbstractAlgebra.Ring) = R
@amalg_ring(Zn::Nemo.NmodRing, Zm::Nemo.NmodRing) =
	ResidueRing(ZZ, gcd(modulus(Zn), modulus(Zm)))

# why is Nemo.NmodRing not a ResRing??
function amalg_rule(::AnyRingOp, k1::FqNmodFiniteField,
		k2::FqNmodFiniteField)
	p = characteristic(k1)
	characteristic(k2) != p && return NullRing
	return Nemo.FiniteField(p, lcm(degree(k1), degree(k2)), "x")
end

# univariate for now
function amalg_rule(op::AnyRingOp, a::PolyRing, b::PolyRing)
	c = amalg_parent(op, base_ring(a), base_ring(b))
	if var(a) == var(b)
		return PolynomialRing(c, var(a))[1]
	else
		return PolynomialRing(c, [var(a), var(b)])[1]
	end
end
function amalg_rule(op::AnyRingOp, a::AbstractAlgebra.PolyRing,
		b::AbstractAlgebra.Ring)
	c = amalg_parent(op, base_ring(a), b)
	return PolynomialRing(c, var(a))[1]
end
# disambiguation
@amalg_ring(a::AbstractAlgebra.PolyRing, ::FlintIntegerRing) = R

# Modules<<<3
function amalg_rule(op::typeof(*), a::AbstractAlgebra.Ring,
		b::AbstractAlgebra.MatSpace)
	c = amalg_parent(op, a, base_ring(b))
	return AbstractAlgebra.MatrixSpace(c, b.nrows, b.ncols)
end
function amalg_rule(op::AnyGroupOp, a::AbstractAlgebra.MatSpace,
		b::AbstractAlgebra.MatSpace)
	c = amalg_parent(op, base_ring(a), base_ring(b))
	@assert a.nrows == b.nrows
	@assert a.ncols == b.ncols
	return AbstractAlgebra.MatrixSpace(c, b.nrows, b.ncols)
end
# Operators<<<2
# function op_by_promote(op, a, b)
# 	R = amalg_rule(op, parent(a), parent(b))
# 	return op(R(a), R(b))
# end

# Nemo.nmod +-* <<<3
@inline Base.:+(x::Nemo.nmod, y::Nemo.nmod) =
	_op_nmod(+, Val(parent(x)==parent(y)), x, y)
@inline Base.:-(x::Nemo.nmod, y::Nemo.nmod) =
	_op_nmod(-, Val(parent(x)==parent(y)), x, y)

function _op_nmod(::typeof(+), ::Val{true}, x::Nemo.nmod, y::Nemo.nmod)
	R = parent(x)
	n = modulus(R)
	d = x.data + y.data - n
	if d > x.data return Nemo.nmod(d+n, R)
	else return Nemo.nmod(d, R)
	end
end
function _op_nmod(::typeof(-), ::Val{true}, x::Nemo.nmod, y::Nemo.nmod)
	R = parent(x)
	n = modulus(R)
	d = x.data - y.data
	if d > x.data return nmod(d + n, R)
	else return nmod(d, R)
	end
end
function _op_nmod(op, ::Val{false}, x::Nemo.nmod, y::Nemo.nmod)
	R = amalg_rule(op, parent(x), parent(y))
	return R(op(x.data, y.data))
end

@inline Base.:*(x::Nemo.nmod, y::Nemo.nmod) =
	_op_nmod(*, Val(parent(x)==parent(y)), x, y)
function _op_nmod(::typeof(*), ::Val{true}, x::Nemo.nmod, y::Nemo.nmod)
	X = parent(x)
	d = ccall((:n_mulmod2_preinv, Nemo.libflint), UInt,
		(UInt, UInt, UInt, UInt), x.data, y.data, X.n, X.ninv)
	return Nemo.nmod(d, X)
end

# Matrices<<<3
@inline (M::MatSpace)(a::MatElem) =
	M([a[i,j] for i in 1:nrows(a), j in 1:ncols(a)])

function Base.:*(a::RingElem, b::ModuleElem)
	new_base_ring = amalg_parent(*, parent(a), base_ring(parent(b)))
	new_mat_space = amalg_parent(*, new_base_ring, parent(b))
	new_base_ring(a)*new_mat_space(b)
end
function op_mat(op::typeof(+), a::MatElem, b::MatElem)
	@assert nrows(a) == nrows(b)
	@assert ncols(a) == ncols(b)
	new_base_ring = amalg_parent(op, base_ring(parent(a)), base_ring(parent(b)))
	new_mat_space = MatrixSpace(new_base_ring, nrows(a), ncols(a))
	# FIXME: this is not broadcasted; it allocates an array and loops are
	# *not* fused:
	new_mat_space([op(a[i,j], b[i,j]) for i in 1:nrows(a), j in 1:ncols(a)])
end
@inline Base.:+(a::MatElem, b::MatElem) = op_mat(+, a, b)
@inline Base.:-(a::MatElem, b::MatElem) = op_mat(-, a, b)

#>>>1
module IP2S #<<<1
using Nemo, LinearAlgebra, AbstractAlgebra
import Nemo: base_ring

# Linear algebra utilities <<<2
# span<<<
"""
		span(A)

Returns a r×n matrix with same column span as `A`,
where `r` == rank(A) and `n` is the size of A.
"""
span(A::M) where{M<:MatElem} =
	let H = hnf(A'), r = rank(H)
	H'[:,1:r]
end#>>>
# as_polynomials <<<
"""
		as_polynomials(A)

Converts the matrix `A` from ring `R` to polynomial ring `R[t]`.
"""
as_polynomials(A::MatElem, t::Symbol = :t) =
	let (m, n) = size(A), R = base_ring(A),
			(RT, T) = PolynomialRing(R, t)
	matrix(RT, [A[i,j] for i in 1:m, j in 1:n])
end #>>>
# pseudoinverse<<<
"""
		pseudoinverse(A)

Given a square matrix `A` with coefficients in a field,
returns a matrix `α` such that `A*α*A == A`.
"""
pseudoinverse(A::MatElem) =
	# Let T.A.U = S, with S in Smith normal form.
	# Since the base ring is a field, S is diagonal(1, …, 1, 0, …, 0)
	# Hence, with α = UST: A α A = T^-1(TAU)S(TAU)U^-1 = T^-1 S^3 U^-1 = A,
	# so that α is an equation-solving inverse of A.
	let (S, T, U) = snf_with_transform(A)
		U*S*T
end #>>>
# # Ax=By solver <<<
# struct CachedSNF{M}
#		A::M
#		S::M; T::M; U::M
#		function CachedSNF(A::M) where {M<:MatElem}
#			(S, T, U) = snf_with_transform(A)
#			new{M}(A, S, T, U)
#		end
# end
# @inline base_ring(S::CachedSNF) = base_ring(S.A)
# """
#			SolveAxBy
# 
# Describes a solver of linear equations of the form `A*x = B*y`
# for fixed matrices `A` and `B` (which allows caching of helper matrices).
# """
# struct SolveAxBy{M}
#		A::CachedSNF{M}
#		B::CachedSNF{M}
#		function SolveAxBy(A::M, B::M) where {M<:MatElem}
#			@assert base_ring(A) == base_ring(B)
#			new{M}(CachedSNF(A), CachedSNF(B))
#		end
# end
# @inline base_ring(S::SolveAxBy) = base_ring(S.A)
# """
#			solve(P::SolveAxBy, Y, {y0})
# 
# Given an affine space of the form `y ∈ y0 + Y`,
# returns a triple `(x0, X, φ)`,
# such that the solutions of `A x = B y` are parametrized by `x ∈ x0 + X`
# and by the relation `y = φ(x)`.
# """
# function solve(P::SolveAxBy{M}, Y::M,
#		y0::M = zero_matrix(base_ring(Y), size(Y,1), 1)) where {M<:MatElem}
#		k = base_ring(P)
#		# Let TA*A*UA = SA, then the relation Ax = By gives
#		# SA UA^-1 x = TA B y
#		# With SA in Smith normal form, this is equivalent to
#		# { y ∈ image(SA); x = UA TA B y + UA SA' Z }
#		# hence the affine space for x is generated
# end
# # >>>
# Pull-back of two matrices
@inline rcef(A::MatElem) = let (d, B) = rref(A'); (d, B') end
@inline identity_left(A::MatElem) = identity_matrix(base_ring(A), size(A, 1))
@inline identity_right(A::MatElem)= identity_matrix(base_ring(A), size(A, 2))

function r_first_coordinates(A::MatElem, r::Integer)
# returns the intersection of colspan(A) and first r coordinates
# as a pair (rank, generating matrix)
	n = size(A, 1)
	k = base_ring(A)
	P = matrix(k, n-r, n, [ i+r == j for i in 1:n-r, j in 1:n ])
	(e, K) = nullspace(P*A)
	(d, L) = rcef(A*K)
	(d, L[1:r, 1:d])
end
# triangular_permute <<<
"""
		triangular_permute(U)

Given an upper-triangular matrix U, a permutation Q
such that UQ has diagonal (1,…,1,0,…,0).
"""
function triangular_permute(U::MatElem)
	(n, m) = size(U)
	P = zeros(Int, m)
	(top, bot) = (1, m)
	for i in 1:m
#			println("examining U[$top, $i]=$(U[top,i])")
		if U[top, i] != 0
			P[i] = top; top+= 1
		else
			P[i] = bot; bot-= 1
		end
	end
	Perm(P)
end
@inline Base.:*(U::MatElem, P::Generic.Perm) = (P*U')'
#>>>
@inline matid(k::AbstractAlgebra.Ring, n::Integer, m::Integer = n) =
	matrix(k, n, m, [ i == j for i in 1:n, j in 1:m ])
function preimage(A::MatElem, Y::MatElem)
# return a parametrization of the space of solutions for AX=Y
	@assert base_ring(A) == base_ring(Y)
	@assert size(A, 1) == size(Y, 1)
	(n, m) = size(A)
	k = base_ring(A)
	(r, p, L, U) = lu(A)
	Q = triangular_permute(U); UQ = U*Q
	dump(Q)
	# we solve the equation UQ.Q^1 X = L^-1PY
	@assert view(UQ, r+1:n, :) == 0
	# it is guaranteed that only the r first rows of UQ are non-zero:
	(r₁, Y₁) = r_first_coordinates(inv(L)*(p*Y), r)
#		@assert r₁ == r
	# with Q^-1 X = [X1; X2]:
	# equation U1 X1 + B X2 = Y1 y, or X1 = - U1^-1 B X2 + U1^-1 Y
	U1 = view(UQ, 1:r, 1:r); U1inv = inv(U1)
	B = view(UQ, 1:r, r+1:m)
	R = Q * [ -U1inv*B	-U1inv*Y₁; matid(k, n-r) zero_matrix(k, n-r, r₁) ]
	R
end
"""
		pullback(A, B, {X}, {Y})

Returns a triple `(Z, φ, ψ)` such that the space of solutions
of `Ax = By` is parametrized by `z ∈ Z; x = φ(z); y = ψ(z)`.
"""
function pullback(A::M, B::M,
	X::M = identity_right(A), Y::M = identity_right(B)) where {M<:MatElem}
	@assert base_ring(A) == base_ring(B)
	@assert size(A, 1) == size(B, 1)
	# This equation is the same as [A -B][x;y] ∈ W= [X 0; 0 Y]
	# with P[A -B] = L*U: L*U*[x;y] ∈ P*W, or U*[x;y] ∈ L^-1*P*W.
	# The vector span of U is V_r = the r first rows;
	# let W' = (L^-1**W) ∩ V_r.
	# Write again U = [U_r C; 0 0] with U_r invertible,
	# then the equation becomes [1_r U_r^-1 C][x;y] ∈ U_r W'.
end
# Bilinear pencils <<<2
abstract type AbstractBilinearPencil end
struct BilinearPencil{R,M} <: AbstractBilinearPencil#<<<
# M is matrix type, T is element type
# M is a true Julia matrix type here...
	base_ring::R
	q0::M
	q∞::M
	BilinearPencil(R::Nemo.AbstractAlgebra.Ring, A::T, B::T) where{T<:MatElem} =
		let sA=size(A), sB=size(B)
		@assert sA[1] == sA[2] == sB[1] == sB[2]
		n = sA[1]; Rnn = MatrixSpace(R, n, n)
		new{typeof(R), elem_type(Rnn)}(R, A, B)
	end

	BilinearPencil(R::Nemo.AbstractAlgebra.Ring, A::T, B::T) where{T<:AbstractMatrix} =
		let sA = size(A), sB = size(B)
#			RA = R.(A), RB = R.(B),
		@assert sA[1] == sA[2] == sB[1] == sB[2]
		n = sA[1]; Rnn = MatrixSpace(R, n, n)
		new{typeof(R), elem_type(Rnn)}(R, Rnn(A), Rnn(B))
	end
end#>>>
# constructors <<<
BilinearPencil(A::T, B::T) where{T<:AbstractMatrix} =
	let R = Nemo.base_ring(A[1,1])
	BilinearPencil(R, R.(A), R.(B))
end

BilinearPencil(A::T, B::T) where{T<:MatElem} =
	let R = Nemo.base_ring(A)
	BilinearPencil(R, A, B)
end
#>>>
# accessors <<<
@inline dim(Π::BilinearPencil) = size(Π.q0, 1)
@inline base_ring(Π::BilinearPencil) = Π.base_ring
@inline q0(Π::BilinearPencil) = Π.q0
@inline q∞(Π::BilinearPencil) = Π.q∞
@inline Base.show(io::IO, Π::BilinearPencil) =
	print(io, "Pencil over $(Π.base_ring):\n$(Π.q0)\n$(Π.q∞)")
@inline vector_space(Π::BilinearPencil) =
	VectorSpace(base_ring(Π), dim(Π))

@inline base_ring_poly(Π::AbstractBilinearPencil, s::Symbol=:t) =
	Nemo.PolynomialRing(base_ring(Π), s)
# >>>

# function transform(Π::AbstractBilinearPencil, P::

function charpoly(Π::AbstractBilinearPencil, t::Symbol=:t)#<<<
#		n = dim(Π)
	RT, T = base_ring_poly(Π, t)
	U = as_polynomials(Π, t)
	Nemo.det(U[1]-T*U[2])
#		RTnn = MatrixSpace(RT, n, n)
#		Nemo.det(RTnn(q0(Π)) - T*RTnn(q∞(Π)))
end#>>>
@inline is_regular(Π::AbstractBilinearPencil) = charpoly(Π) != 0
transform(Π::BilinearPencil, P::MatElem) = begin
	@assert base_ring(P) == base_ring(Π)
	@assert size(P, 1) == dim(Π)
	BilinearPencil(P'*q0(Π)*P, P'*q∞(Π)*P)
end
# kronecker_sequence<<<
"""
		kronecker_sequence(A, B, F, c)

`A` and `B` being `n`×`n` square matrices,
`F` being a `r`×`n` matrix, and `c` being a column of length `r`,
return a parametrization of the solutions of equations:

		F*A*y0 = c; A y1 = B y0; A y2 = B y1; …

The solutions are returned as a vector of pairs of matrices (Y\\_i, φ\\_i),
such that:\n
	A Y\\_0 = 0;\n
	A Y\\_i = B Y\\_{i-1} = B φ\\_i Y\\_i.

`c` is either a `FieldElem` or a (Julia) constant (e.g. zero).
"""
kronecker_sequence(A::TM, B::TM,
	F::TM = one(parent(A)),
	c::TM = zero_matrix(base_ring(A), size(F,1), 1)) where {TM<:MatElem} = begin
	print("c=$c\n")
#			A y\\_i = B y\\_{i-1}; F A y\\_0 = c.
	n = size(A, 1)
	@assert size(A) == (n,n) && size(B) == (n,n)
	k = Nemo.base_ring(A)
	@assert k isa AbstractAlgebra.Field
	@assert Nemo.base_ring(B) == Nemo.base_ring(A)
	# The first equation is F A x1 = c.
	# We return the triple (x1p, X1, φ=0)
	# such that x1 = x1a + X1*whatever.
	L = let (S, T, U) = snf_with_transform(F*A),
					p = size(F, 2), r = rank(S),
					S₁ = vcat(zero_matrix(k, r, n-r), identity_matrix(k, n-r)),
					KerFA = span(U*S₁)
#		println("S=$S, S1=$(S₁)")
#		println("FA=$(F*A), kernel=$(KerFA)\n$(nullspace(F*A)[2])")
#		println("c=$c, U*S'*T*c = $(T*c)")
		[ (U*S'*T*c, KerFA, zero_matrix(k, 0, size(KerFA, 2))) ]
	end
	@assert F*A*L[1][1] == c
	@assert F*A*L[1][2] == 0
	# Next equations are A x2 = B x1, etc.
	# this means that x2 = α B x1 + x'2,
	#		where α is a pseudoinverse of A, and x'2 ∈ Ker A.
	# If x1 = x1a + X1*…, then
	#		x2 = αB x1a + hcat(αB X1, Ker A)*….
	# We then find a suitable x1 from x2 in the same way:
	# x1 = βA x2, β = pseudoinverse of B.
	β = pseudoinverse(B)
	α = pseudoinverse(A)
	display(nullspace(β*A*α*B)[2])
	return
	KerA= nullspace(A)[2]

	for i in 1:n-1
		let (x1a, X1, φ1) = last(L)
			X2 = (hcat(α*B*X1, KerA))
			x2a= α*B*x1a
			println("i=$i\n----")
			@assert A*x2a == B*x1a
			@assert A*α*B*X1 == B*X1
			println("A*X2=")
			display(A*X2)
			println("B*X1=")
			display(B*X1)
			# the sequence will eventually stabilize (for dimension reasons),
			# no need to go on forever...
			(X2 == X1) && break
			push!(L, (x2a, X2, β*A))
		end
	end
	L
end
#>>>
# kronecker_factor <<<
"""
		kronecker_factor(Π::AbstractBilinearPencil)

Returns a triple `(d, P, Q)` such that:\n

 - P ⊕ Q is a orthogonal decomposition
 - dim(P) = 2d+1
 - P ∘ Π is isomorphic to the Kronecker module K_d

If this is not possible, then return the triple (-1, undef, undef)
instead.

Implements the algorithm from [Waterhouse 1976, theorem 3.1].

"""
function kronecker_factor(Π::AbstractBilinearPencil)
	A=q0(Π); B=q∞(Π); n = dim(Π)
	k = base_ring(Π)
	(RT, T) = base_ring_poly(Π)
	(AT, BT) = (as_polynomials(A), as_polynomials(B))
	c = Nemo.det(AT-T*BT)
	if c != 0
		return (-1, undef, undef)
	end

	L = kronecker_sequence(A, B)

	shortest_chain() = for m in 1:length(L)
		(e, K) = nullspace(B*L[m][2])
		e ≥ 1 && return (m, e, K)
	end
	(m, e, K) = shortest_chain()

	println("shortest_chain: ($m, $e, $K)")
	display(L[4][3]*L[4][2])
	display(L[3][2])
	return
	v = Vector{typeof(A)}(undef, m); v[m] = L[m][2]*K[:,1]
	for i = m:-1:2
		v[i-1] = L[i][3]*v[i]
	end
	display(L[4][2]*matrix(k,4,1,[1,0,0,1])==v[4])
	return
#		println(L[4][2]*matrix(k,1,4,[0,1,0,1]))
	for i = 1:m
		println("v[$i]=$(v[i])")
		println("v[$i]-L[$i][1]=$(v[i]-L[i][1])")
		println("L[$i][2]=$(L[i][2]')")
	end
	@assert A*v[1] == 0
	@assert all([A*v[i] == B*v[i-1] for i = 2:m-1])
	@assert B*v[m] == 0

	L1 = kronecker_sequence(B, A, transpose(v[1])*B,
		matrix(k, 1, 1, [one(k)]))
	# L1[i] parametrizes the space of xn such that
	return L1
#		# compute the kernel K and find a vector with minimum degree
#			K = nullspace(Q0-T*Qi)[2]
#			for j in 1:size(K,2)
#				d = content(K[:,j])
#				for i in 1:size(K,1)
#					K[i,j] = divexact(K[i,j], d)
#				end
#			end
#			println((Q0-T*Qi)*v)
#			# now find a vector in K with minimum degree
#			return K
#				F =  keys(factor(c).fac), # list of nmod_polys
#				# FIXME do something if F is empty
#				fmin = partialsort([F...],1, by=degree), # lowest-degree factor
#		 z=0
#		 (Q0, Qi)
end
#>>>

end # module
#>>>1

st(x)=supertypes(typeof(x))
stp(x)=st(parent(x))
macro st(x) :(st($x)) end
macro stp(x) :(stp($x)) end

k = FF(3)
RT,t = k[:t]
q0 = matrix(k, [1 2; 0 1])
# using .IP2S
# 
# k = FF(3)
# R_3 = k^(3,3)
# # Z2_4 = Nemo.MatrixSpace(Z2,4,4)
# # Z2_4 = Nemo.MatrixSpace(Z2,4,4)
# A4 = [0 1 1 0; 0 1 0 1; 1 0 0 0 ; 0 1 1 1]
# # B = [0 0 1 1; 0 0 0 1; 0 1 1 0 ; 1 1 0 0]
K1(n,s) = [ i==j+s for i = 1:n+1, j = 1:n ]
K(n,s) = [ zeros(Int,n+1,n+1) K1(n,s); K1(n,s)' zeros(Int, n, n)]
# 
# # A = [0 0 0; 0 0 1; 0 1 0]
# # B = [0 0 1; 0 0 0; 1 0 0]
# # M = [0 0 1 0 2; 2 1 1 2 1; 2 2 2 2 2; 1 1 2 1 0; 0 0 0 0 1]
# # H = [0 -1 1 3; 1 0 0 0; 0 2 0 0; 0 0 1 0; 0 0 0 1]
n = 3
M=[1 0 1 1 1 0 1;0 0 1 1 1 1 0;0 1 0 1 0 1 0;0 1 1 1 1 0 1;1 0 1 1 0 1 0;1 0 0 0 0 1 0;1 1 0 0 0 1 0]
A = M'*K(n,0)*M; RA=matrix(k,A)
B = M'*K(n,1)*M; RB=matrix(k,B)
# P = IP2S.BilinearPencil(k, A, B)



# vim: fmr=<<<,>>> noet ts=2
