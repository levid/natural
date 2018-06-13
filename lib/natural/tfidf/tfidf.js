/*
Copyright (c) 2011, Rob Ellis, Chris Umbel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

var _ = require("underscore")._,
  Tokenizer = require("../tokenizers/regexp_tokenizer").WordTokenizer,
  tokenizer = new Tokenizer(),
  stopwords = require("../util/stopwords").words,
  fs = require("fs");

var similarity = require("compute-cosine-similarity");

function buildDocument(text, key) {
  var stopOut;

  if (typeof text === "string") {
    text = tokenizer.tokenize(text.toLowerCase());
    stopOut = true;
  } else if (!_.isArray(text)) {
    stopOut = false;
    return text;
  }

  return text.reduce(
    function(document, term) {
      // next line solves https://github.com/NaturalNode/natural/issues/119
      if (typeof document[term] === "function") document[term] = 0;
      if (!stopOut || stopwords.indexOf(term) < 0)
        document[term] = document[term] ? document[term] + 1 : 1;
      return document;
    },
    { __key: key }
  );
}

function tf(term, document) {
  return document[term] ? document[term] : 0;
}

function documentHasTerm(term, document) {
  return document[term] && document[term] > 0;
}

function TfIdf(deserialized) {
  if (deserialized) this.documents = deserialized.documents;
  else this.documents = [];

  this._idfCache = {};
}

// backwards compatibility for < node 0.10
function isEncoding(encoding) {
  if (typeof Buffer.isEncoding !== "undefined")
    return Buffer.isEncoding(encoding);
  switch ((encoding + "").toLowerCase()) {
    case "hex":
    case "utf8":
    case "utf-8":
    case "ascii":
    case "binary":
    case "base64":
    case "ucs2":
    case "ucs-2":
    case "utf16le":
    case "utf-16le":
    case "raw":
      return true;
  }
  return false;
}

module.exports = TfIdf;
TfIdf.tf = tf;

TfIdf.prototype.idf = function(term, force) {
  // Lookup the term in the New term-IDF caching,
  // this will cut search times down exponentially on large document sets.
  if (
    this._idfCache[term] &&
    this._idfCache.hasOwnProperty(term) &&
    force !== true
  )
    return this._idfCache[term];

  var docsWithTerm = this.documents.reduce(function(count, document) {
    return count + (documentHasTerm(term, document) ? 1 : 0);
  }, 0);

  var idf = 1 + Math.log(this.documents.length / (1 + docsWithTerm));

  // Add the idf to the term cache and return it
  this._idfCache[term] = idf;
  return idf;
};

TfIdf.prototype.tfidfVector = function(terms, d) {
  var _this = this;

  if (!_.isArray(terms))
    terms = tokenizer.tokenize(terms.toString().toLowerCase());

  return terms.map(function(term) {
    var idf = _this.idf(term);
    idf = idf === Infinity ? 0 : idf;
    return tf(term, _this.documents[d]) * idf;
  });
};

// returns document array, sorted by cosine similarity to the terms
TfIdf.prototype.tfidfsCosine = function(terms, callback) {
  // build tfidf terms vector:
  // [ tfidf(term1, terms), ... , tfidf(termN, terms) ]
  var termDoc = buildDocument(terms);
  var termVec = [];
  var numDocuments = 1;
  var docsWithTerm = 1;
  for (var term in termDoc) {
    if ("__key" === term) continue;
    var idf = 1 + Math.log(numDocuments / (1 + docsWithTerm));
    termVec.push(tf(term, termDoc) * idf);
  }

  // build tfidf terms vector for each doc
  var tfidfsVecs = new Array(this.documents.length);

  for (var i = 0; i < this.documents.length; i++) {
    tfidfsVecs[i] = this.tfidfVector(terms, i);
    tfidfsVecs[i].cosine = similarity(termVec, tfidfsVecs[i]);
    tfidfsVecs[i].i = i;

    if (callback) callback(i, tfidfsVecs[i], this.documents[i].__key);
  }

  var sorted = tfidfsVecs.sort(function(a, b) {
    return b.cosine - a.cosine;
  });
  return sorted;
};

// If restoreCache is set to true, all terms idf scores currently cached will be recomputed.
// Otherwise, the cache will just be wiped clean
TfIdf.prototype.addDocument = function(document, key, restoreCache) {
  this.documents.push(buildDocument(document, key));

  // make sure the cache is invalidated when new documents arrive
  if (restoreCache === true) {
    for (var term in this._idfCache) {
      // invoking idf with the force option set will
      // force a recomputation of the idf, and it will
      // automatically refresh the cache value.
      this.idf(term, true);
    }
  } else {
    this._idfCache = {};
  }
};

// If restoreCache is set to true, all terms idf scores currently cached will be recomputed.
// Otherwise, the cache will just be wiped clean
TfIdf.prototype.addFileSync = function(path, encoding, key, restoreCache) {
  if (!encoding) encoding = "utf8";
  if (!isEncoding(encoding)) throw new Error("Invalid encoding: " + encoding);

  var document = fs.readFileSync(path, encoding);
  this.documents.push(buildDocument(document, key));

  // make sure the cache is invalidated when new documents arrive
  if (restoreCache === true) {
    for (var term in this._idfCache) {
      // invoking idf with the force option set will
      // force a recomputation of the idf, and it will
      // automatically refresh the cache value.
      this.idf(term, true);
    }
  } else {
    this._idfCache = {};
  }
};

TfIdf.prototype.tfidf = function(terms, d) {
  var _this = this;

  if (!_.isArray(terms)) {
    terms = tokenizer.tokenize(terms.toString().toLowerCase());
  }

  return terms.reduce(function(value, term) {
    var idf = _this.idf(term);
    idf = idf === Infinity ? 0 : idf;
    return value + tf(term, _this.documents[d]) * idf;
  }, 0.0);
};

TfIdf.prototype.listTerms = function(d) {
  var terms = [];
  var _this = this;
  for (var term in this.documents[d]) {
    if (this.documents[d]) {
      if (term != "__key") {
        terms.push({
          term: term,
          tf: tf(term, _this.documents[d]),
          idf: _this.idf(term),
          tfidf: _this.tfidf(term, d)
        });
      }
    }
  }

  return terms.sort(function(x, y) {
    return y.tfidf - x.tfidf;
  });
};

TfIdf.prototype.tfidfs = function(terms, callback) {
  var tfidfs = new Array(this.documents.length);

  for (var i = 0; i < this.documents.length; i++) {
    tfidfs[i] = this.tfidf(terms, i);

    if (callback) callback(i, tfidfs[i], this.documents[i].__key);
  }

  return tfidfs;
};

// Define a tokenizer other than the default "WordTokenizer"
TfIdf.prototype.setTokenizer = function(t) {
  if (!_.isFunction(t.tokenize)) throw new Error("Expected a valid Tokenizer");
  tokenizer = t;
};

// Define a stopwords other than the default
TfIdf.prototype.setStopwords = function(customStopwords) {
  if (!Array.isArray(customStopwords)) return false;

  customStopwords.forEach(stopword => {
    if (typeof stopword != "string") return false;
  });

  stopwords = customStopwords;
  return true;
};
