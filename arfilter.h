#pragma once

template< class T >
class ARFilter
{
public:
    ARFilter( double alpha = 0.01 )
        :_alpha(alpha), _init(false)
    {

    }

    virtual ~ARFilter()
    {

    }

    T update( T v )
    {
        // T ret = _alpha * (_lastOutput) + (T(1.0 - _alpha)) * v;
        if( !_init )
        {
            _lastOutput = v;
            _init = true;
        }

        T ret = _lastOutput - _alpha * (_lastOutput - v);
        _lastOutput = ret;
        return ret;
    }

    T& alpha()
    {
        return _alpha;
    }

private:
    T _lastOutput;
    double _alpha;
    bool _init;

protected:

};
