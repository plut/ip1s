module Flint # Flint wrappers
return
import Base: ==, <, cmp, isless, iszero, isone, iseven, isodd
import Base: +, -, *, fld, mod, div, rem, abs, muladd
import Base: log, isqrt, gcd, lcm, invmod
import Base: trailing_zeros, sign, bitstring, count_ones
import Base: parse, string, show
import Base: size, getindex, setindex!, iterate, vcat, hcat
export issquare, Fmpz, @fmpz_str, FmpzMat

# Generalities <<<1
const FlintLong = Int64
const libflint = "libflint"
const _c_signed=filter(t->sizeof(t) <= sizeof(FlintLong),
	[Int8,Int16,Int32,Int64,Int128])

const Csigned = Union{_c_signed...}
const Cunsigned=Union{Bool,unsigned.(_c_signed)...}

"""
    _flint(fname=>rettype, arg1=>type1, arg2=>type2, ...)

Wrapper around `ccall` in `"libflint"`.
Argument types may be omitted in the following case:
 - `Ref{Fmpz}` for `Fmpz`;
 - `Clong` for signed int types <= `Clong`;
 - `Culong` for unsigned int types <= `Culong`.
"""
@inline function _flint((fname,rettype), args...)
	q = quote ccall(($(QuoteNode(fname)), libflint), $rettype,
		($(_flint_c_type.(args)...),),
		$(_flint_c_val.(args)...))
	end
	return eval(q)
end

_flint_c_type(::T) where{T<:Ptr} = T
_flint_c_type(::Csigned) = FlintLong
_flint_c_type(::Cunsigned) = unsigned(FlintLong)
_flint_c_type(::BigInt) = Ref{BigInt}
_flint_c_type(p::Pair{<:Any,<:Type}) = p[2]
# _flint_c_type for Fmp*** follows once those types are defined

_flint_c_val(x::Any) = x
_flint_c_val(p::Pair{<:Any,<:Type}) = p[1]


function flint_string(p::Ptr{Cchar})
	s = unsafe_string(p)
	_flint(:flint_free=>Nothing, p=>Ptr{Cchar})
	return s
end

# Fmpz <<<1
# Type and constructors<<<2
mutable struct Fmpz <: Signed
	data::FlintLong # signed long
	@inline Fmpz(fname::Symbol, args...) =
		_set(finalizer(fmpz_clear, new()), fname, args...)
	@inline Fmpz() = Fmpz(:fmpz_init)
	@inline Fmpz(x::Csigned) = Fmpz(:fmpz_init_set_si, x)
	@inline Fmpz(x::Cunsigned) = Fmpz(:fmpz_init_set_ui, x)
	@inline Fmpz(x::Fmpz) = Fmpz(:fmpz_init_set, x)
end
_flint_c_type(::Fmpz) = Ref{Fmpz}

@inline function _set(z, fname::Symbol, args...)
	_flint(fname=>Nothing, z, args...)
	return z
end
@inline _set(r::Ptr{Fmpz}, z::Fmpz) = _set(r, :fmpz_set, z)
@inline _set(r::Ptr{Fmpz}, z::Csigned) = _set(r, :fmpz_set_si, z)
@inline _set(r::Ptr{Fmpz}, z::Cunsigned) = _set(r, :fmpz_set_ui, z)

@inline fmpz_clear(z::Fmpz) = _flint(:fmpz_clear=>Nothing, z)

# Conversion to mpz<<<2
@inline Fmpz(x::BigInt) = _set(Fmpz(), :fmpz_set_mpz, x)
@inline zero(::Type{Fmpz}) = _set(Fmpz(), :fmpz_zero)
@inline one(::Type{Fmpz}) = _set(Fmpz(), :fmpz_one)

function parse(::Type{Fmpz}, s::AbstractString; base::Int=10)
	@assert base >= 2
	@assert base <= 62
	z = Fmpz()
	r = _flint(:fmpz_set_str=>Cint, z, s=>Ptr{Cchar}, base=>Cint)
	@assert r == 0
	return z
end
macro fmpz_str(s)
	return parse(Fmpz, s; base=10)
end

# Conversion from mpz<<<2
@inline is_big(z::Fmpz) = (z.data & (1<<(8*sizeof(z.data)-2)))

(::Type{FlintLong})(z::Fmpz) =
	_flint(:fmpz_get_si=>FlintLong, z)
(::Type{unsigned(FlintLong)})(z::Fmpz) =
	_flint(:fmpz_get_ui=>unsigned(FlintLong), z::Fmpz)
@inline function (::Type{BigInt})(z::Fmpz)
	x = BigInt()
	_flint(:fmpz_get_mpz=>Nothing, x, z)
	return x
end

function string(z::Fmpz; base::Integer=10)
	@assert base >= 2
	@assert base <= 62
	return flint_string(_flint(:fmpz_get_str=>Ptr{Cchar}, C_NULL=>Ptr{Cchar},
		base=>Cint, z))
end

@inline show(io::IO, z::Fmpz) = print(io, string(z; base=10))
# Basic properties and manipulation<<<2

@inline sign(z::Fmpz) = _flint(:fmpz_sgn=>Cint, z)
@inline trailing_zeros(z::Fmpz) = _flint(:fmpz_val2=>Cint, z)

# Comparison
@inline cmp(z::Fmpz, w::Fmpz) = _flint(:fmpz_cmp=>Cint, z, w=>Ref{Fmpz})
@inline cmp(z::Fmpz, w::Csigned) = _flint(:fmpz_cmp_si=>Cint, z, w)
@inline cmp(z::Fmpz, w::Cunsigned) = _flint(:fmpz_cmp_ui=>Cint, z, w)
@inline cmp(z::Integer, w::Fmpz) = -cmp(w,z)

@inline isless(z::Fmpz, w::Integer) = cmp(z,w) < 0
@inline isless(z::Integer, w::Fmpz) = cmp(z,w) < 0
@inline isless(z::Fmpz, w::Fmpz) = cmp(z,w) < 0

# Base.:< uses promotion, which we want to avoid here.
@inline <(z::Fmpz, w::Integer) = cmp(z,w) < 0
@inline <(z::Integer, w::Fmpz) = cmp(z,w) < 0
@inline <(z::Fmpz, w::Fmpz) = cmp(z,w) < 0

@inline ==(z::Fmpz, w::Integer) = cmp(z,w) == 0
@inline ==(z::Integer, w::Fmpz) = cmp(z,w) == 0
@inline ==(z::Fmpz, w::Fmpz) = cmp(z,w) == 0


@inline iszero(z::Fmpz)::Bool = _flint(:fmpz_is_zero=>Cint, z)
@inline isone(z::Fmpz)::Bool = _flint(:fmpz_is_one=>Cint, z)
@inline iseven(z::Fmpz)::Bool = _flint(:fmpz_is_even=>Cint, z)
@inline isodd(z::Fmpz)::Bool = _flint(:fmpz_is_odd=>Cint, z)

# Basic arithmetic<<<2
@inline -(z::Fmpz) = _set(Fmpz(), :fmpz_neg, z)
@inline abs(z::Fmpz) = _set(Fmpz(), :fmpz_abs, z)

@inline +(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_add, z, w)
@inline +(z::Fmpz, w::Cunsigned) = _set(Fmpz(), :fmpz_add_ui, z, w)
@inline +(w::Cunsigned, z::Fmpz) = z+w

@inline -(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_sub, z, w)
@inline -(z::Fmpz, w::Cunsigned) = _set(Fmpz(), :fmpz_sub_ui, z, w)

@inline +(z::Fmpz, w::Csigned) = (w >= 0) ? z+unsigned(w) : z-unsigned(w)
@inline +(w::Csigned, z::Fmpz) = z+w
@inline -(z::Fmpz, w::Csigned) = z+(-w)
@inline -(w::Csigned, z::Fmpz) = z+(-w)

@inline *(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_mul, z, w)
@inline *(z::Fmpz, w::Csigned) = _set(Fmpz(), :fmpz_mul_si, z, w)
@inline *(z::Fmpz, w::Cunsigned) = _set(Fmpz(), :fmpz_mul_ui, z, w)
@inline *(w::Csigned, z::Fmpz) = z*w
@inline *(w::Cunsigned, z::Fmpz) = z*w
# TODO: uiui

# FIXME: decide what the best argument order is for this function
@inline function muladd!(z::Fmpz, x::Fmpz, y::Fmpz)
	_flint(:fmpz_addmul=>Nothing, z, x, y) # z ← z + x y
	return z
end
@inline muladd(x::Fmpz, y::Fmpz, z::Fmpz) = muladd!(Fmpz(z), x, y) # xy+z
@inline function muladd!(z::Fmpz, x::Fmpz, y::Cunsigned)
	_flint(:fmpz_addmul_ui=>Nothing, z, x, y)
	return z
end
@inline muladd(x::Fmpz, y::Csigned, z::Fmpz) = muladd!(Fmpz(z), x, y)
@inline muladd(x::Csigned, y::Fmpz, z::Fmpz) = muladd!(Fmpz(z), y, x)

@inline fld(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_fdiv_q, z, w)
@inline fld(z::Fmpz, w::Csigned) = _set(Fmpz(), :fmpz_fdiv_q_si, z, w)
@inline fld(z::Fmpz, w::Cunsigned) = _set(Fmpz(), :fmpz_fdiv_q_ui, z, w)
@inline div(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_tdiv_q, z, w)
@inline div(z::Fmpz, w::Csigned) = _set(Fmpz(), :fmpz_tdiv_q_si, z, w)
@inline div(z::Fmpz, w::Cunsigned) = _set(Fmpz(), :fmpz_tdiv_q_ui, z, w)
@inline mod(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_mod, z, w)
@inline mod(z::Fmpz, w::Cunsigned) = _flint(:fmpz_fdiv_ui=>Culong, z, w)

@inline log(z::Fmpz) = _flint(:fmpz_dlog=>Cdouble, z)
@inline isqrt(z::Fmpz) = _set(Fmpz(), :fmpz_sqrt, z)
@inline issquare(z::Fmpz) = _flint(:fmpz_is_square=>Cint, z)

# GCD<<<2
@inline gcd(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_gcd, z, w)
@inline lcm(z::Fmpz, w::Fmpz) = _set(Fmpz(), :fmpz_lcm, z, w)
@inline function invmod(z::Fmpz, w::Fmpz)
	y = Fmpz()
	r = _flint(:fmpz_invmod=>Cint, y, z, w)
	@assert r != 0 "z is not invertible mod w"
	return y
end

# Logic<<<2
@inline bitstring(z::Fmpz) = string(z; base=2)
@inline count_ones(z::Fmpz) = _flint(:fmpz_popcnt=>Cint, z)

# FmpzMat<<<1
# Type, constructors, accessors<<<2
mutable struct FmpzMat <: AbstractMatrix{Fmpz}
	entries::Ptr{Nothing}
	r::FlintLong
	c::FlintLong
	rows::Ptr{Nothing}

	function FmpzMat(fname::Symbol, args...)
		m = finalizer(fmpz_mat_clear, new())
		_flint(fname=>Nothing, m, args...)
		return m
	end
end
_flint_c_type(::FmpzMat) = Ref{FmpzMat}
@inline FmpzMat(rows::Csigned, cols::Csigned) =
	FmpzMat(:fmpz_mat_init, rows, cols)
@inline FmpzMat(w::FmpzMat) = FmpzMat(:fmpz_mat_init_set, w)
@inline function Base.getindex(m::FmpzMat, i::Csigned, j::Csigned)
	r = _flint(:fmpz_mat_entry=>Ptr{Fmpz}, m, i-1, j-1)
	return _set(Fmpz(), :fmpz_set, r)
end
@inline function Base.setindex!(m::FmpzMat, z::Integer, i::Csigned, j::Csigned)
	r = _flint(:fmpz_mat_entry=>Ptr{Fmpz}, m, i-1, j-1)
	return _set(r, z)
end
@inline size(m::FmpzMat) = (m.r, m.c)
function Base.iterate(m::FmpzMat, state=(1,1))
	(state[2] > m.c) && return nothing
	newstate = (state[1] > m.r) ? (1, state[2]+1) : (state[1]+1, state[2])
	return (a[state...], newstate)
end

@inline fmpz_mat_clear(m::FmpzMat) = _flint(:fmpz_mat_clear=>Nothing, m)
@inline similar(::FmpzMat, ::Fmpz, r::Int, c::Int) = FmpzMat(r, c)

# TODO: typed_hvcat to build Fmpz[...]
# Comparison<<<2
@inline ==(m::FmpzMat, w::FmpzMat)::Bool = _flint(:fmpz_mat_equal=>Cint, m, w)
@inline iszero(m::FmpzMat)::Bool = _flint(:fmpz_mat_is_zero=>Cint, m)
@inline isempty(m::FmpzMat)::Bool = _flint(:fmpz_mat_is_empty=>Cint, m)
@inline issquare(m::FmpzMat)::Bool = _flint(:fmpz_mat_is_square=>Cint, m)

# Concatenation<<<2
function vcat(m1::FmpzMat, m2::FmpzMat)
	@assert m1.c == m2.c "m1 and m2 have same number of columns"
	return _set(FmpzMat(m1.r + m2.r, m1.c), :fmpz_mat_concat_vertical, m1, m2)
end
function hcat(m1::FmpzMat, m2::FmpzMat)
	@assert m1.r == m2.r "m1 and m2 have same number of rows"
	return _set(FmpzMat(m1.r, m1.c + m2.c), :fmpz_mat_concat_horizontal, m1, m2)
end
@inline vcat(m::FmpzMat...) = reduce(vcat, m)
@inline hcat(m::FmpzMat...) = reduce(hcat, m)
# Additions <<<2
# Matrix.scalar <<<2
# Matrix multiplication<<<2
# >>>1
end

using .Flint
Nothing

# vim: ts=2 noet fmr=<<<,>>>:
